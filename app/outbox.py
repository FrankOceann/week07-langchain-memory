from collections.abc import Callable
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import and_, or_, select, update

from app.models import (
    MemoryOutbox,
    OUTBOX_STATUS_FAILED,
    OUTBOX_STATUS_PENDING,
    OUTBOX_STATUS_PROCESSING,
    OUTBOX_STATUS_SUCCEEDED,
)


class MemoryOutboxRepository:
    def __init__(self, session_factory: Callable):
        self.session_factory = session_factory

    def claim_next(
        self,
        now: datetime,
        lease_seconds: int = 60,
    ) -> MemoryOutbox | None:
        statement = (
            select(MemoryOutbox)
            .where(
                or_(
                    and_(
                        MemoryOutbox.status == OUTBOX_STATUS_PENDING,
                        MemoryOutbox.available_at <= now,
                    ),
                    and_(
                        MemoryOutbox.status == OUTBOX_STATUS_PROCESSING,
                        MemoryOutbox.lease_expires_at < now,
                    ),
                )
            )
            .order_by(MemoryOutbox.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )

        with self.session_factory() as session:
            event = session.scalar(statement)

            if event is None:
                return None

            event.status = OUTBOX_STATUS_PROCESSING
            event.attempt_count += 1
            event.lease_token = str(uuid4())
            event.lease_expires_at = now + timedelta(
                seconds=lease_seconds
            )
            session.commit()

            return event

    def mark_succeeded(
        self,
        event_id: int,
        lease_token: str,
        now: datetime,
    ) -> bool:
        statement = (
            update(MemoryOutbox)
            .where(
                MemoryOutbox.id == event_id,
                MemoryOutbox.status == OUTBOX_STATUS_PROCESSING,
                MemoryOutbox.lease_token == lease_token,
            )
            .values(
                status=OUTBOX_STATUS_SUCCEEDED,
                processed_at=now,
                lease_token=None,
                lease_expires_at=None,
                last_error=None,
            )
        )

        with self.session_factory() as session:
            result = session.execute(statement)
            session.commit()

            return result.rowcount == 1

    def mark_failed(
        self,
        event_id: int,
        lease_token: str,
        error: Exception,
        now: datetime,
        max_attempts: int = 3,
    ) -> str:
        statement = (
            select(MemoryOutbox)
            .where(
                MemoryOutbox.id == event_id,
                MemoryOutbox.status == OUTBOX_STATUS_PROCESSING,
                MemoryOutbox.lease_token == lease_token,
            )
            .with_for_update(skip_locked=True)
        )

        with self.session_factory() as session:
            event = session.scalar(statement)

            if event is None:
                return "lost_lease"

            event.last_error = (
                f"{type(error).__name__}: {error}"[:1000]
            )
            event.lease_token = None
            event.lease_expires_at = None

            if event.attempt_count >= max_attempts:
                event.status = OUTBOX_STATUS_FAILED
                session.commit()
                return "failed"

            event.status = OUTBOX_STATUS_PENDING
            event.available_at = now + timedelta(
                seconds=2 ** (event.attempt_count - 1)
            )
            session.commit()

            return "retrying"

    def retry_all_failed(self, now: datetime) -> int:
        statement = (
            update(MemoryOutbox)
            .where(MemoryOutbox.status == OUTBOX_STATUS_FAILED)
            .values(
                status=OUTBOX_STATUS_PENDING,
                attempt_count=0,
                available_at=now,
                lease_token=None,
                lease_expires_at=None,
                last_error=None,
                processed_at=None,
            )
        )

        with self.session_factory() as session:
            result = session.execute(statement)
            session.commit()

            return result.rowcount


class MemoryOutboxWorker:
    def __init__(
        self,
        outbox_repository,
        long_term_memory_repository,
        memory_sync_service,
        vector_index,
        max_attempts: int = 3,
    ):
        self.outbox_repository = outbox_repository
        self.long_term_memory_repository = (
            long_term_memory_repository
        )
        self.memory_sync_service = memory_sync_service
        self.vector_index = vector_index
        self.max_attempts = max_attempts

    def drain(
        self,
        limit: int,
        now: datetime | None = None,
    ) -> dict[str, int]:
        if limit < 1:
            raise ValueError("limit 必须至少为 1。")

        current_time = now or datetime.now()
        result = {"succeeded": 0, "retrying": 0, "failed": 0}

        for _ in range(limit):
            event = self.outbox_repository.claim_next(current_time)

            if event is None:
                break

            try:
                memory = self.long_term_memory_repository.get_by_id(
                    event.memory_id
                )

                if memory is not None and memory.is_active:
                    self.memory_sync_service.sync(memory)
                else:
                    self.vector_index.delete(event.memory_id)

                if self.outbox_repository.mark_succeeded(
                    event.id,
                    event.lease_token,
                    current_time,
                ):
                    result["succeeded"] += 1
            except Exception as error:
                outcome = self.outbox_repository.mark_failed(
                    event.id,
                    event.lease_token,
                    error,
                    current_time,
                    max_attempts=self.max_attempts,
                )

                if outcome in result:
                    result[outcome] += 1

        return result

from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import (
    Base,
    MemoryOutbox,
    OUTBOX_STATUS_FAILED,
    OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
    OUTBOX_STATUS_PENDING,
    OUTBOX_STATUS_PROCESSING,
    OUTBOX_STATUS_SUCCEEDED,
)
from app.outbox import MemoryOutboxRepository, MemoryOutboxWorker


def test_claim_next_claims_due_pending_event_with_lease():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add_all(
            [
                MemoryOutbox(
                    memory_id=101,
                    event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                    status=OUTBOX_STATUS_PENDING,
                    available_at=now,
                ),
                MemoryOutbox(
                    memory_id=102,
                    event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                    status=OUTBOX_STATUS_PENDING,
                    available_at=now + timedelta(seconds=1),
                ),
            ]
        )
        session.commit()

    repository = MemoryOutboxRepository(session_factory)

    event = repository.claim_next(now)

    assert event is not None
    assert event.memory_id == 101
    assert event.status == OUTBOX_STATUS_PROCESSING
    assert event.attempt_count == 1
    assert event.lease_token is not None
    assert event.lease_expires_at == now + timedelta(seconds=60)
    assert repository.claim_next(now) is None


def test_claim_next_reclaims_only_expired_processing_event():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add_all(
            [
                MemoryOutbox(
                    memory_id=101,
                    event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                    status=OUTBOX_STATUS_PROCESSING,
                    attempt_count=1,
                    available_at=now,
                    lease_token="expired-token",
                    lease_expires_at=now - timedelta(seconds=1),
                ),
                MemoryOutbox(
                    memory_id=102,
                    event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                    status=OUTBOX_STATUS_PROCESSING,
                    attempt_count=1,
                    available_at=now,
                    lease_token="valid-token",
                    lease_expires_at=now + timedelta(seconds=1),
                ),
            ]
        )
        session.commit()

    repository = MemoryOutboxRepository(session_factory)

    event = repository.claim_next(now)

    assert event is not None
    assert event.memory_id == 101
    assert event.attempt_count == 2
    assert event.lease_token != "expired-token"
    assert repository.claim_next(now) is None


def test_mark_succeeded_completes_event_owned_by_lease_token():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add(
            MemoryOutbox(
                memory_id=101,
                event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                status=OUTBOX_STATUS_PENDING,
                available_at=now,
            )
        )
        session.commit()

    repository = MemoryOutboxRepository(session_factory)
    event = repository.claim_next(now)

    assert event is not None
    assert repository.mark_succeeded(
        event.id,
        event.lease_token,
        now,
    ) is True

    with session_factory() as session:
        completed_event = session.get(MemoryOutbox, event.id)

    assert completed_event.status == OUTBOX_STATUS_SUCCEEDED
    assert completed_event.processed_at == now
    assert completed_event.lease_token is None
    assert completed_event.lease_expires_at is None


def test_mark_failed_requeues_first_failure_with_backoff():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add(
            MemoryOutbox(
                memory_id=101,
                event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                status=OUTBOX_STATUS_PENDING,
                available_at=now,
            )
        )
        session.commit()

    repository = MemoryOutboxRepository(session_factory)
    event = repository.claim_next(now)

    assert event is not None
    assert repository.mark_failed(
        event.id,
        event.lease_token,
        RuntimeError("Milvus 暂时不可用"),
        now,
    ) == "retrying"

    with session_factory() as session:
        failed_event = session.get(MemoryOutbox, event.id)

    assert failed_event.status == OUTBOX_STATUS_PENDING
    assert failed_event.attempt_count == 1
    assert failed_event.available_at == now + timedelta(seconds=1)
    assert failed_event.lease_token is None
    assert failed_event.lease_expires_at is None
    assert failed_event.last_error == "RuntimeError: Milvus 暂时不可用"


def test_mark_failed_marks_third_attempt_as_failed():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    first_now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add(
            MemoryOutbox(
                memory_id=101,
                event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                status=OUTBOX_STATUS_PENDING,
                available_at=first_now,
            )
        )
        session.commit()

    repository = MemoryOutboxRepository(session_factory)

    first_event = repository.claim_next(first_now)
    assert first_event is not None
    assert repository.mark_failed(
        first_event.id,
        first_event.lease_token,
        RuntimeError("第一次失败"),
        first_now,
    ) == "retrying"

    second_now = first_now + timedelta(seconds=1)
    second_event = repository.claim_next(second_now)
    assert second_event is not None
    assert repository.mark_failed(
        second_event.id,
        second_event.lease_token,
        RuntimeError("第二次失败"),
        second_now,
    ) == "retrying"

    third_now = second_now + timedelta(seconds=2)
    third_event = repository.claim_next(third_now)
    assert third_event is not None
    assert repository.mark_failed(
        third_event.id,
        third_event.lease_token,
        RuntimeError("第三次失败"),
        third_now,
    ) == "failed"

    with session_factory() as session:
        failed_event = session.get(MemoryOutbox, third_event.id)

    assert failed_event.status == OUTBOX_STATUS_FAILED
    assert failed_event.attempt_count == 3
    assert failed_event.lease_token is None
    assert failed_event.lease_expires_at is None
    assert failed_event.last_error == "RuntimeError: 第三次失败"


def test_retry_all_failed_requeues_only_failed_events():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add_all(
            [
                MemoryOutbox(
                    memory_id=101,
                    event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                    status=OUTBOX_STATUS_FAILED,
                    attempt_count=3,
                    available_at=now - timedelta(seconds=1),
                    lease_token="old-token",
                    lease_expires_at=now - timedelta(seconds=1),
                    last_error="RuntimeError: Milvus 暂时不可用",
                    processed_at=now - timedelta(seconds=1),
                ),
                MemoryOutbox(
                    memory_id=102,
                    event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                    status=OUTBOX_STATUS_PENDING,
                    available_at=now,
                ),
            ]
        )
        session.commit()

    repository = MemoryOutboxRepository(session_factory)

    assert repository.retry_all_failed(now) == 1

    with session_factory() as session:
        requeued_event = session.get(MemoryOutbox, 1)
        pending_event = session.get(MemoryOutbox, 2)

    assert requeued_event.status == OUTBOX_STATUS_PENDING
    assert requeued_event.attempt_count == 0
    assert requeued_event.available_at == now
    assert requeued_event.lease_token is None
    assert requeued_event.lease_expires_at is None
    assert requeued_event.last_error is None
    assert requeued_event.processed_at is None
    assert pending_event.status == OUTBOX_STATUS_PENDING


def test_worker_syncs_active_memory_and_completes_event():
    class ActiveMemoryRepository:
        def __init__(self, memory):
            self.memory = memory

        def get_by_id(self, memory_id):
            assert memory_id == self.memory.id
            return self.memory

    class RecordingMemorySyncService:
        def __init__(self):
            self.memories = []

        def sync(self, memory):
            self.memories.append(memory)

    class RecordingVectorIndex:
        def __init__(self):
            self.deleted_memory_ids = []

        def delete(self, memory_id):
            self.deleted_memory_ids.append(memory_id)

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add(
            MemoryOutbox(
                memory_id=101,
                event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                status=OUTBOX_STATUS_PENDING,
                available_at=now,
            )
        )
        session.commit()

    memory = SimpleNamespace(
        id=101,
        user_id="frank",
        content="回答时优先使用中文。",
        is_active=True,
    )
    outbox_repository = MemoryOutboxRepository(session_factory)
    memory_repository = ActiveMemoryRepository(memory)
    memory_sync_service = RecordingMemorySyncService()
    vector_index = RecordingVectorIndex()
    worker = MemoryOutboxWorker(
        outbox_repository=outbox_repository,
        long_term_memory_repository=memory_repository,
        memory_sync_service=memory_sync_service,
        vector_index=vector_index,
    )

    result = worker.drain(limit=1, now=now)

    assert result == {
        "succeeded": 1,
        "retrying": 0,
        "failed": 0,
    }
    assert memory_sync_service.memories == [memory]
    assert vector_index.deleted_memory_ids == []

    with session_factory() as session:
        event = session.get(MemoryOutbox, 1)

    assert event.status == OUTBOX_STATUS_SUCCEEDED


def test_worker_deletes_vector_for_inactive_memory():
    class InactiveMemoryRepository:
        def get_by_id(self, memory_id):
            assert memory_id == 101
            return SimpleNamespace(
                id=101,
                user_id="frank",
                content="已停用的偏好。",
                is_active=False,
            )

    class SyncServiceThatMustNotRun:
        def sync(self, memory):
            raise AssertionError("停用记忆不应生成新向量。")

    class RecordingVectorIndex:
        def __init__(self):
            self.deleted_memory_ids = []

        def delete(self, memory_id):
            self.deleted_memory_ids.append(memory_id)

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add(
            MemoryOutbox(
                memory_id=101,
                event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                status=OUTBOX_STATUS_PENDING,
                available_at=now,
            )
        )
        session.commit()

    outbox_repository = MemoryOutboxRepository(session_factory)
    vector_index = RecordingVectorIndex()
    worker = MemoryOutboxWorker(
        outbox_repository=outbox_repository,
        long_term_memory_repository=InactiveMemoryRepository(),
        memory_sync_service=SyncServiceThatMustNotRun(),
        vector_index=vector_index,
    )

    result = worker.drain(limit=1, now=now)

    assert result == {
        "succeeded": 1,
        "retrying": 0,
        "failed": 0,
    }
    assert vector_index.deleted_memory_ids == [101]

    with session_factory() as session:
        event = session.get(MemoryOutbox, 1)

    assert event.status == OUTBOX_STATUS_SUCCEEDED


def test_worker_retries_event_when_memory_sync_fails():
    class ActiveMemoryRepository:
        def get_by_id(self, memory_id):
            assert memory_id == 101
            return SimpleNamespace(
                id=101,
                user_id="frank",
                content="回答时优先使用中文。",
                is_active=True,
            )

    class FailingMemorySyncService:
        def sync(self, memory):
            raise ConnectionError("embedding timeout")

    class VectorIndexThatMustNotRun:
        def delete(self, memory_id):
            raise AssertionError("有效记忆不应走删除路径。")

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    now = datetime(2026, 8, 31, 12, 0, 0)

    with session_factory() as session:
        session.add(
            MemoryOutbox(
                memory_id=101,
                event_type=OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
                status=OUTBOX_STATUS_PENDING,
                available_at=now,
            )
        )
        session.commit()

    outbox_repository = MemoryOutboxRepository(session_factory)
    worker = MemoryOutboxWorker(
        outbox_repository=outbox_repository,
        long_term_memory_repository=ActiveMemoryRepository(),
        memory_sync_service=FailingMemorySyncService(),
        vector_index=VectorIndexThatMustNotRun(),
    )

    result = worker.drain(limit=1, now=now)

    assert result == {
        "succeeded": 0,
        "retrying": 1,
        "failed": 0,
    }

    with session_factory() as session:
        event = session.get(MemoryOutbox, 1)

    assert event.status == OUTBOX_STATUS_PENDING
    assert event.attempt_count == 1
    assert event.available_at == now + timedelta(seconds=1)
    assert event.last_error == "ConnectionError: embedding timeout"

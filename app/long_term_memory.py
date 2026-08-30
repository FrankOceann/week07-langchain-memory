from collections.abc import Callable

from sqlalchemy import select

from app.models import LongTermMemory


ALLOWED_CATEGORIES = {"preference", "profile", "fact"}


class SQLAlchemyLongTermMemoryRepository:
    def __init__(self, session_factory: Callable):
        self.session_factory = session_factory

    def add(
        self,
        user_id: str,
        category: str,
        content: str,
        source: str = "user_confirmed",
    ) -> LongTermMemory:
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(
                "category 必须是 preference、profile 或 fact。"
            )

        if not user_id.strip() or not content.strip():
            raise ValueError("user_id 和 content 不能为空。")

        with self.session_factory() as session:
            memory = LongTermMemory(
                user_id=user_id.strip(),
                category=category,
                content=content.strip(),
                source=source,
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)

            return memory

    def list_active(
        self,
        user_id: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[LongTermMemory]:
        if category is not None and category not in ALLOWED_CATEGORIES:
            raise ValueError(
                "category 必须是 preference、profile 或 fact。"
            )

        if limit < 1:
            raise ValueError("limit 必须至少为 1。")

        statement = select(LongTermMemory).where(
            LongTermMemory.user_id == user_id,
            LongTermMemory.is_active.is_(True),
        )

        if category is not None:
            statement = statement.where(
                LongTermMemory.category == category
            )

        statement = statement.order_by(
            LongTermMemory.updated_at.desc(),
            LongTermMemory.id.desc(),
        ).limit(limit)

        with self.session_factory() as session:
            return list(session.scalars(statement))

    def list_active_by_ids(
        self,
        user_id: str,
        memory_ids: list[int],
    ) -> list[LongTermMemory]:
        if not memory_ids:
            return []

        statement = select(LongTermMemory).where(
            LongTermMemory.id.in_(memory_ids),
            LongTermMemory.user_id == user_id,
            LongTermMemory.is_active.is_(True),
        )

        with self.session_factory() as session:
            return list(session.scalars(statement))

    def deactivate(self, memory_id: int) -> bool:
        with self.session_factory() as session:
            memory = session.get(LongTermMemory, memory_id)

            if memory is None or not memory.is_active:
                return False

            memory.is_active = False
            session.commit()

            return True


def render_long_term_memories(
    memories: list[LongTermMemory],
) -> str:
    if not memories:
        return "无已确认长期记忆。"

    return "\n".join(
        f"[memory:{memory.id}] ({memory.category}) {memory.content}"
        for memory in memories
    )
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.long_term_memory import (
    SQLAlchemyLongTermMemoryRepository,
    render_long_term_memories,
)
from app.models import (
    Base,
    MemoryOutbox,
    OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
    OUTBOX_STATUS_PENDING,
)


@pytest.fixture
def repository():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    return SQLAlchemyLongTermMemoryRepository(
        sessionmaker(
            bind=engine,
            expire_on_commit=False,
        )
    )

def test_repository_add_creates_pending_outbox_event():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    repository = SQLAlchemyLongTermMemoryRepository(session_factory)

    memory = repository.add(
        "frank",
        "preference",
        "回答时优先使用中文。",
    )

    with session_factory() as session:
        events = list(
            session.scalars(
                select(MemoryOutbox).order_by(MemoryOutbox.id)
            )
        )

    assert len(events) == 1
    event = events[0]
    assert event.memory_id == memory.id
    assert event.event_type == OUTBOX_EVENT_MEMORY_INDEX_REQUESTED
    assert event.status == OUTBOX_STATUS_PENDING
    assert event.attempt_count == 0
    assert event.available_at is not None
    assert event.lease_token is None
    assert event.lease_expires_at is None
    assert event.last_error is None
    assert event.processed_at is None

def test_repository_deactivate_creates_pending_outbox_event():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    repository = SQLAlchemyLongTermMemoryRepository(session_factory)

    memory = repository.add(
        "frank",
        "preference",
        "回答时优先使用中文。",
    )

    assert repository.deactivate(memory.id) is True

    with session_factory() as session:
        events = list(
            session.scalars(
                select(MemoryOutbox).order_by(MemoryOutbox.id)
            )
        )

    assert len(events) == 2
    assert [event.memory_id for event in events] == [
        memory.id,
        memory.id,
    ]
    assert [event.event_type for event in events] == [
        OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
        OUTBOX_EVENT_MEMORY_INDEX_REQUESTED,
    ]
    assert [event.status for event in events] == [
        OUTBOX_STATUS_PENDING,
        OUTBOX_STATUS_PENDING,
    ]

def test_repository_get_by_id_returns_inactive_memory_for_worker(
    repository,
):
    memory = repository.add(
        "frank",
        "preference",
        "回答时优先使用中文。",
    )

    assert repository.deactivate(memory.id) is True

    found = repository.get_by_id(memory.id)

    assert found is not None
    assert found.id == memory.id
    assert found.user_id == "frank"
    assert found.is_active is False
    assert repository.get_by_id(999) is None

def test_repository_isolates_users_and_hides_deactivated_memory(
    repository,
):
    old = repository.add("frank", "preference", "旧偏好")
    repository.add("frank", "profile", "正在学习 AI Agent")
    repository.add("alice", "preference", "Alice 的偏好")

    assert repository.deactivate(old.id) is True

    memories = repository.list_active("frank")

    assert [(item.user_id, item.content) for item in memories] == [
        ("frank", "正在学习 AI Agent"),
    ]


def test_repository_filters_category_limits_and_rejects_invalid_input(
    repository,
):
    repository.add("frank", "preference", "偏好一")
    repository.add("frank", "profile", "资料一")
    repository.add("frank", "preference", "偏好二")

    memories = repository.list_active(
        "frank",
        category="preference",
        limit=1,
    )

    assert [item.content for item in memories] == ["偏好二"]

    with pytest.raises(ValueError, match="category"):
        repository.add("frank", "unknown", "内容")

    assert repository.deactivate(999) is False

def test_render_long_term_memories_marks_empty_and_includes_memory_id(
    repository,
):
    assert render_long_term_memories([]) == "无已确认长期记忆。"

    memory = repository.add("frank", "preference", "使用中文")

    assert render_long_term_memories([memory]) == (
        f"[memory:{memory.id}] (preference) 使用中文"
    )

def test_repository_reads_only_current_users_active_candidate_ids(
    repository,
):
    active = repository.add("frank", "preference", "有效记忆")
    inactive = repository.add("frank", "profile", "已停用记忆")
    other_user = repository.add("alice", "fact", "Alice 的记忆")

    assert repository.deactivate(inactive.id) is True

    memories = repository.list_active_by_ids(
        "frank",
        [other_user.id, inactive.id, active.id, 999],
    )

    assert [(memory.id, memory.user_id, memory.content) for memory in memories] == [
        (active.id, "frank", "有效记忆"),
    ]
import pytest
from sqlalchemy import inspect

from app.database import build_engine, get_mysql_url
from app.models import Base, LongTermMemory


def test_get_mysql_url_rejects_missing_url(monkeypatch):
    monkeypatch.delenv("MYSQL_URL", raising=False)

    with pytest.raises(ValueError, match="MYSQL_URL"):
        get_mysql_url(mysql_url="")


def test_long_term_memory_model_declares_expected_columns():
    engine = build_engine("sqlite+pysqlite:///:memory:")
    LongTermMemory.metadata.create_all(engine)

    columns = {
        column["name"]
        for column in inspect(engine).get_columns("long_term_memories")
    }

    assert columns == {
        "id",
        "user_id",
        "category",
        "content",
        "source",
        "is_active",
        "created_at",
        "updated_at",
    }


def test_memory_outbox_model_declares_ready_event_schema():
    engine = build_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    columns = {
        column["name"]
        for column in inspector.get_columns("memory_outbox")
    }

    assert columns == {
        "id",
        "memory_id",
        "event_type",
        "status",
        "attempt_count",
        "available_at",
        "lease_token",
        "lease_expires_at",
        "last_error",
        "processed_at",
        "created_at",
        "updated_at",
    }
    assert inspector.get_indexes("memory_outbox") == [
        {
            "name": "ix_memory_outbox_status_available_id",
            "column_names": ["status", "available_at", "id"],
            "unique": 0,
            "dialect_options": {},
        }
    ]

import pytest
from sqlalchemy import inspect

from app.database import build_engine, get_mysql_url
from app.models import LongTermMemory


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
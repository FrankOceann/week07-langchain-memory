import fakeredis
import pytest

from langchain_core.messages import AIMessage, HumanMessage

from app.memory import (
    RedisHistoryStore,
    SessionHistoryStore,
    build_redis_client,
)


def test_session_history_store_keeps_sessions_isolated():
    store = SessionHistoryStore(max_turns=3)

    store.get("session-a").add_messages(
        [HumanMessage("A 问题"), AIMessage("A 回答")]
    )
    store.get("session-b").add_messages(
        [HumanMessage("B 问题"), AIMessage("B 回答")]
    )

    assert [item.content for item in store.get("session-a").messages] == [
        "A 问题",
        "A 回答",
    ]
    assert [item.content for item in store.get("session-b").messages] == [
        "B 问题",
        "B 回答",
    ]

def test_session_history_store_keeps_only_latest_complete_turns():
    store = SessionHistoryStore(max_turns=2)
    history = store.get("session-a")

    for index in range(3):
        history.add_messages(
            [HumanMessage(f"Q{index}"), AIMessage(f"A{index}")]
        )

    assert [item.content for item in history.messages] == [
        "Q1",
        "A1",
        "Q2",
        "A2",
    ]

def test_redis_history_keeps_sessions_isolated_and_trims_turns():
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisHistoryStore(
        client,
        max_turns=2,
        ttl_seconds=30,
    )

    for index in range(3):
        store.get("session-a").add_messages(
            [HumanMessage(f"A-Q{index}"), AIMessage(f"A-A{index}")]
        )

    store.get("session-b").add_messages(
        [HumanMessage("B-Q0"), AIMessage("B-A0")]
    )

    assert [item.content for item in store.get("session-a").messages] == [
        "A-Q1",
        "A-A1",
        "A-Q2",
        "A-A2",
    ]
    assert [item.content for item in store.get("session-b").messages] == [
        "B-Q0",
        "B-A0",
    ]


def test_redis_history_refreshes_ttl_and_clear_only_removes_its_session():
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisHistoryStore(client, max_turns=3, ttl_seconds=30)
    history_a = store.get("session-a")
    history_b = store.get("session-b")

    history_a.add_messages([HumanMessage("A-Q"), AIMessage("A-A")])
    history_b.add_messages([HumanMessage("B-Q"), AIMessage("B-A")])

    assert client.ttl(history_a.key) > 0

    history_a.clear()

    assert history_a.messages == []
    assert [item.content for item in history_b.messages] == ["B-Q", "B-A"]


def test_build_redis_client_rejects_missing_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(ValueError, match="REDIS_URL"):
        build_redis_client(redis_url="")
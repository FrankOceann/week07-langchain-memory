from langchain_core.messages import AIMessage, HumanMessage

from app.memory import SessionHistoryStore


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
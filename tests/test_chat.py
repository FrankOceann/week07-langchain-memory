from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest
from app.chat import (
    ask_question,
    build_chat_model,
    build_conversation_runnable,
)
from app.memory import SessionHistoryStore


class FakeRetriever:
    def invoke(self, question: str) -> list[Document]:
        return [
            Document(
                page_content="确认副作用前必须征得用户同意。",
                metadata={"source": "agent_safety.txt#chunk-0"},
            )
        ]


def test_same_session_includes_previous_turn_and_preserves_sources():
    received_prompts = []

    def fake_response(prompt_value):
        received_prompts.append(prompt_value.messages)
        return AIMessage(content="假的回答")

    conversation_runnable = build_conversation_runnable(
        RunnableLambda(fake_response),
        SessionHistoryStore(max_turns=3),
    )

    first_answer, first_sources = ask_question(
        "如何确认副作用操作？",
        session_id="session-a",
        retriever=FakeRetriever(),
        conversation_runnable=conversation_runnable,
    )
    second_answer, second_sources = ask_question(
        "那为什么？",
        session_id="session-a",
        retriever=FakeRetriever(),
        conversation_runnable=conversation_runnable,
    )

    second_prompt_text = "\n".join(
    message.content for message in received_prompts[1]
    )

    assert first_answer == "假的回答"
    assert second_answer == "假的回答"
    assert first_sources == ["agent_safety.txt#chunk-0"]
    assert second_sources == ["agent_safety.txt#chunk-0"]
    assert "如何确认副作用操作？" in second_prompt_text
    assert "假的回答" in second_prompt_text
    assert "确认副作用前必须征得用户同意。" in second_prompt_text

def test_build_chat_model_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        build_chat_model(api_key="")
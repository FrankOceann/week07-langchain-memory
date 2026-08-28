from types import SimpleNamespace
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest
import fakeredis
from app.chat import (
    ask_question,
    build_chat_model,
    build_conversation_runnable,
)
from app.memory import RedisHistoryStore


class FakeRetriever:
    def invoke(self, question: str) -> list[Document]:
        return [
            Document(
                page_content="确认副作用前必须征得用户同意。",
                metadata={"source": "agent_safety.txt#chunk-0"},
            )
        ]

class FakeLongTermMemoryRepository:
    def __init__(self):
        self.requested_user_ids = []

    def list_active(self, user_id: str, limit: int = 5):
        self.requested_user_ids.append(user_id)

        if user_id == "frank":
            return [
                SimpleNamespace(
                    id=101,
                    category="preference",
                    content="使用中文回答。",
                )
            ]

        return []


def test_same_session_includes_previous_turn_and_preserves_sources():
    received_prompts = []

    def fake_response(prompt_value):
        received_prompts.append(prompt_value.messages)
        return AIMessage(content="假的回答")

    conversation_runnable = build_conversation_runnable(
        RunnableLambda(fake_response),
        RedisHistoryStore(
            fakeredis.FakeRedis(decode_responses=True),
            max_turns=3,
            ttl_seconds=30,
        ),
    )
    long_term_memory_repository = FakeLongTermMemoryRepository()

    first_answer, first_sources = ask_question(
        "如何确认副作用操作？",
        session_id="session-a",
        user_id="existing-user",
        retriever=FakeRetriever(),
        conversation_runnable=conversation_runnable,
        long_term_memory_repository=long_term_memory_repository,
    )
    second_answer, second_sources = ask_question(
        "那为什么？",
        session_id="session-a",
        user_id="existing-user",
        retriever=FakeRetriever(),
        conversation_runnable=conversation_runnable,
        long_term_memory_repository=long_term_memory_repository,
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

def test_chat_includes_current_users_long_term_memories():
    received_prompts = []

    def fake_response(prompt_value):
        received_prompts.append(prompt_value.messages)
        return AIMessage(content="假的回答")

    repository = FakeLongTermMemoryRepository()
    conversation_runnable = build_conversation_runnable(
        RunnableLambda(fake_response),
        RedisHistoryStore(
            fakeredis.FakeRedis(decode_responses=True),
            max_turns=3,
            ttl_seconds=30,
        ),
    )

    answer, sources = ask_question(
        "如何确认副作用操作？",
        session_id="session-a",
        user_id="frank",
        retriever=FakeRetriever(),
        conversation_runnable=conversation_runnable,
        long_term_memory_repository=repository,
    )

    prompt_text = "\n".join(
        message.content for message in received_prompts[0]
    )

    assert answer == "假的回答"
    assert sources == ["agent_safety.txt#chunk-0"]
    assert repository.requested_user_ids == ["frank"]
    assert "[memory:101] (preference) 使用中文回答。" in prompt_text

def test_long_term_memory_failure_does_not_call_chat_model():
    class FailingLongTermMemoryRepository:
        def list_active(self, user_id: str, limit: int = 5):
            raise ConnectionError("MySQL 不可用")

    class RecordingConversationRunnable:
        def __init__(self):
            self.calls = 0

        def invoke(self, values, config):
            self.calls += 1
            return AIMessage(content="不应生成")

    conversation_runnable = RecordingConversationRunnable()

    with pytest.raises(ConnectionError, match="MySQL 不可用"):
        ask_question(
            "如何确认副作用操作？",
            session_id="session-a",
            user_id="frank",
            retriever=FakeRetriever(),
            conversation_runnable=conversation_runnable,
            long_term_memory_repository=FailingLongTermMemoryRepository(),
        )

    assert conversation_runnable.calls == 0
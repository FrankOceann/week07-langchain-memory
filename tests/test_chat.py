from types import SimpleNamespace
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest
import fakeredis
from app.chat import (
    build_chat_model,
    ask_question_with_workflow,
)
from app.memory import RedisHistoryStore
from langgraph.checkpoint.memory import InMemorySaver
from app.workflow import build_chat_workflow_graph

class FakeRetriever:
    def invoke(self, question: str) -> list[Document]:
        return [
            Document(
                page_content="确认副作用前必须征得用户同意。",
                metadata={"source": "agent_safety.txt#chunk-0"},
            )
        ]

class FakeSemanticLongTermMemoryService:
    def __init__(self):
        self.search_calls = []

    def search_active(
        self,
        user_id: str,
        question: str,
        limit: int = 3,
    ):
        self.search_calls.append(
            {
                "user_id": user_id,
                "question": question,
                "limit": limit,
            }
        )

        if user_id == "frank":
            return [
                SimpleNamespace(
                    id=101,
                    category="preference",
                    content="使用中文回答。",
                )
            ]

        return []


def test_build_chat_model_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        build_chat_model(api_key="")

def test_workflow_adapter_returns_answer_sources_and_saves_history():
    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )

    def fake_response(prompt_value):
        return AIMessage(content="图工作流回答")

    graph = build_chat_workflow_graph(
        history_store=history_store,
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticLongTermMemoryService(),
        chat_model=RunnableLambda(fake_response),
        checkpointer=InMemorySaver(),
    )

    answer, sources = ask_question_with_workflow(
        question="如何确认副作用操作？",
        session_id="session-a",
        user_id="frank",
        workflow_graph=graph,
    )

    messages = history_store.get("session-a").messages

    assert answer == "图工作流回答"
    assert sources == ["agent_safety.txt#chunk-0"]
    assert [message.content for message in messages] == [
        "如何确认副作用操作？",
        "图工作流回答",
    ]

def test_workflow_adapter_raises_error_without_saving_history():
    class FailingSemanticMemoryService:
        def search_active(
            self,
            user_id: str,
            question: str,
            limit: int = 3,
        ):
            raise ConnectionError("long-term memory unavailable")

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = build_chat_workflow_graph(
        history_store=history_store,
        retriever=FakeRetriever(),
        semantic_memory_service=FailingSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: AIMessage(content="不应生成")
        ),
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(
        RuntimeError,
        match="ConnectionError: long-term memory unavailable",
    ):
        ask_question_with_workflow(
            question="如何确认副作用操作？",
            session_id="session-a",
            user_id="frank",
            workflow_graph=graph,
        )

    assert history_store.get("session-a").messages == []
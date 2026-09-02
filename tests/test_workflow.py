from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from app.workflow import build_minimal_graph
from langchain_core.documents import Document
import app.workflow as workflow
from types import SimpleNamespace
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
import fakeredis
from app.memory import RedisHistoryStore

def test_workflow_processes_nonempty_question():
    graph = build_minimal_graph(InMemorySaver())

    result = graph.invoke(
        {"question": "第一问"},
        config={
            "configurable": {
                "thread_id": "week08:frank:session-a",
            }
        },
    )

    assert result["answer"] == "已处理：第一问"
    assert result["completed_questions"] == ["第一问"]

def test_workflow_routes_empty_question_to_error():
    graph = build_minimal_graph(InMemorySaver())

    result = graph.invoke(
        {"question": "   "},
        config={
            "configurable": {
                "thread_id": "week08:frank:empty-question",
            }
        },
    )

    assert result.get("error") == "问题不能为空。"
    assert result.get("answer") is None

def test_workflow_reuses_checkpoint_for_same_thread():
    graph = build_minimal_graph(InMemorySaver())
    config = {
        "configurable": {
            "thread_id": "week08:frank:session-checkpoint",
        }
    }

    graph.invoke({"question": "第一问"}, config=config)
    result = graph.invoke({"question": "第二问"}, config=config)

    assert result["completed_questions"] == ["第一问", "第二问"]

def test_workflow_interrupts_when_approval_is_required():
    graph = build_minimal_graph(InMemorySaver())

    result = graph.invoke(
        {
            "question": "需要人工审批的问题",
            "requires_approval": True,
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:approval",
            }
        },
    )

    interrupts = result.get("__interrupt__")

    assert interrupts is not None
    assert interrupts[0].value == {
        "action": "process_question",
        "question": "需要人工审批的问题",
    }
    assert result.get("answer") is None

def test_workflow_processes_question_after_approval():
    graph = build_minimal_graph(InMemorySaver())
    config = {
        "configurable": {
            "thread_id": "week08:frank:approval-resume",
        }
    }

    graph.invoke(
        {
            "question": "需要人工审批的问题",
            "requires_approval": True,
        },
        config=config,
    )

    result = graph.invoke(
        Command(resume="approved"),
        config=config,
    )

    assert result.get("answer") == "已处理：需要人工审批的问题"
    assert result.get("completed_questions") == [
        "需要人工审批的问题"
    ]

def test_workflow_stops_when_approval_is_rejected():
    graph = build_minimal_graph(InMemorySaver())
    config = {
        "configurable": {
            "thread_id": "week08:frank:approval-reject",
        }
    }

    graph.invoke(
        {
            "question": "需要人工审批的问题",
            "requires_approval": True,
        },
        config=config,
    )

    result = graph.invoke(
        Command(resume="rejected"),
        config=config,
    )

    assert result.get("error") == "人工审批未通过。"
    assert result.get("answer") is None

class FakeRetriever:
    def __init__(self):
        self.questions = []

    def invoke(self, question: str):
        self.questions.append(question)

        return [
            Document(
                page_content="确认副作用前必须征得用户同意。",
                metadata={"source": "agent_safety.txt#chunk-0"},
            ),
            Document(
                page_content="工具执行前需要校验参数。",
                metadata={"source": "agent_safety.txt#chunk-1"},
            ),
        ]


def test_rag_graph_builds_context_and_sources_from_retriever():
    retriever = FakeRetriever()
    graph = workflow.build_rag_retrieval_graph(
        retriever=retriever,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"question": "如何确认副作用操作？"},
        config={
            "configurable": {
                "thread_id": "week08:frank:rag-retrieval",
            }
        },
    )

    assert retriever.questions == ["如何确认副作用操作？"]
    assert result["sources"] == [
        "agent_safety.txt#chunk-0",
        "agent_safety.txt#chunk-1",
    ]
    assert result["context"] == (
        "[agent_safety.txt#chunk-0]\n"
        "确认副作用前必须征得用户同意。\n\n"
        "[agent_safety.txt#chunk-1]\n"
        "工具执行前需要校验参数。"
    )

class FailingRetriever:
    def invoke(self, question: str):
        raise ConnectionError("retriever unavailable")


def test_rag_graph_records_retrieval_error():
    graph = workflow.build_rag_retrieval_graph(
        retriever=FailingRetriever(),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"question": "如何确认副作用操作？"},
        config={
            "configurable": {
                "thread_id": "week08:frank:rag-error",
            }
        },
    )

    assert result.get("error") == (
        "ConnectionError: retriever unavailable"
    )

class FakeSemanticMemoryService:
    def __init__(self):
        self.calls = []

    def search_active(
        self,
        user_id: str,
        question: str,
        limit: int = 3,
    ):
        self.calls.append(
            {
                "user_id": user_id,
                "question": question,
                "limit": limit,
            }
        )

        return [
            SimpleNamespace(
                id=101,
                category="preference",
                content="使用中文回答。",
            )
        ]


def test_rag_memory_graph_loads_current_users_long_term_memory():
    retriever = FakeRetriever()
    semantic_memory_service = FakeSemanticMemoryService()
    graph = workflow.build_rag_memory_graph(
        retriever=retriever,
        semantic_memory_service=semantic_memory_service,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "question": "如何确认副作用操作？",
            "user_id": "frank",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:rag-memory",
            }
        },
    )

    assert semantic_memory_service.calls == [
        {
            "user_id": "frank",
            "question": "如何确认副作用操作？",
            "limit": 3,
        }
    ]
    assert result["long_term_memory_context"] == (
        "[memory:101] (preference) 使用中文回答。"
    )

def test_rag_memory_graph_skips_long_term_memory_when_retrieval_fails():
    semantic_memory_service = FakeSemanticMemoryService()
    graph = workflow.build_rag_memory_graph(
        retriever=FailingRetriever(),
        semantic_memory_service=semantic_memory_service,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "question": "如何确认副作用操作？",
            "user_id": "frank",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:rag-memory-error",
            }
        },
    )

    assert result.get("error") == (
        "ConnectionError: retriever unavailable"
    )
    assert semantic_memory_service.calls == []

class FailingSemanticMemoryService:
    def search_active(
        self,
        user_id: str,
        question: str,
        limit: int = 3,
    ):
        raise ConnectionError("long-term memory unavailable")


def test_rag_memory_graph_records_long_term_memory_error():
    graph = workflow.build_rag_memory_graph(
        retriever=FakeRetriever(),
        semantic_memory_service=FailingSemanticMemoryService(),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "question": "如何确认副作用操作？",
            "user_id": "frank",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:long-term-memory-error",
            }
        },
    )

    assert result.get("error") == (
        "ConnectionError: long-term memory unavailable"
    )

def test_rag_memory_answer_graph_generates_answer_from_context():
    received_prompts = []

    def fake_response(prompt_value):
        received_prompts.append(prompt_value.messages)
        return AIMessage(content="假的回答")

    graph = workflow.build_rag_memory_answer_graph(
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(fake_response),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "question": "如何确认副作用操作？",
            "user_id": "frank",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:rag-memory-answer",
            }
        },
    )

    prompt_text = "\n".join(
        message.content
        for message in received_prompts[0]
    )

    assert result["answer"] == "假的回答"
    assert "如何确认副作用操作？" in prompt_text
    assert "确认副作用前必须征得用户同意。" in prompt_text
    assert "[memory:101] (preference) 使用中文回答。" in prompt_text

def test_rag_memory_answer_graph_does_not_call_model_after_memory_error():
    chat_calls = []

    def fake_response(prompt_value):
        chat_calls.append(prompt_value)
        return AIMessage(content="不应生成")

    graph = workflow.build_rag_memory_answer_graph(
        retriever=FakeRetriever(),
        semantic_memory_service=FailingSemanticMemoryService(),
        chat_model=RunnableLambda(fake_response),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "question": "如何确认副作用操作？",
            "user_id": "frank",
        },
        config={
            "configurable": {
                "thread_id": (
                    "week08:frank:rag-memory-answer-error"
                ),
            }
        },
    )

    assert result.get("error") == (
        "ConnectionError: long-term memory unavailable"
    )
    assert chat_calls == []

def test_rag_memory_answer_graph_records_model_error():
    def failing_response(prompt_value):
        raise ConnectionError("chat model unavailable")

    graph = workflow.build_rag_memory_answer_graph(
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(failing_response),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "question": "如何确认副作用操作？",
            "user_id": "frank",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:chat-model-error",
            }
        },
    )

    assert result.get("error") == (
        "ConnectionError: chat model unavailable"
    )

def test_chat_history_graph_loads_redis_messages_for_session():
    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    history_store.get("session-a").add_messages(
        [
            HumanMessage(content="上一轮问题"),
            AIMessage(content="上一轮回答"),
        ]
    )

    graph = workflow.build_chat_history_graph(
        history_store=history_store,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"session_id": "session-a"},
        config={
            "configurable": {
                "thread_id": "week08:frank:session-a",
            }
        },
    )

    assert [message.content for message in result["history"]] == [
        "上一轮问题",
        "上一轮回答",
    ]

def test_chat_history_graph_saves_successful_turn_to_redis():
    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_history_graph(
        history_store=history_store,
        checkpointer=InMemorySaver(),
    )

    graph.invoke(
        {
            "session_id": "session-a",
            "question": "本轮问题",
            "answer": "本轮回答",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:session-a",
            }
        },
    )

    messages = history_store.get("session-a").messages

    assert [message.content for message in messages] == [
        "本轮问题",
        "本轮回答",
    ]

def test_chat_history_graph_does_not_resave_previous_turn_after_error():
    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_history_graph(
        history_store=history_store,
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {
            "thread_id": "week08:frank:history-error",
        }
    }

    graph.invoke(
        {
            "session_id": "session-a",
            "question": "成功问题",
            "answer": "成功回答",
        },
        config=config,
    )
    graph.invoke(
        {
            "session_id": "session-a",
            "error": "模型调用失败",
        },
        config=config,
    )

    messages = history_store.get("session-a").messages

    assert [message.content for message in messages] == [
        "成功问题",
        "成功回答",
    ]

def test_chat_workflow_graph_runs_rag_memory_and_redis_history():
    received_prompts = []

    def fake_response(prompt_value):
        received_prompts.append(prompt_value.messages)
        return AIMessage(content="假的回答")

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    history_store.get("session-a").add_messages(
        [
            HumanMessage(content="上一轮问题"),
            AIMessage(content="上一轮回答"),
        ]
    )

    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(fake_response),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:session-a",
            }
        },
    )

    prompt_text = "\n".join(
        message.content
        for message in received_prompts[0]
    )
    messages = history_store.get("session-a").messages

    assert result["answer"] == "假的回答"
    assert result["sources"] == [
    "agent_safety.txt#chunk-0",
    "agent_safety.txt#chunk-1",
    ]
    assert "上一轮问题" in prompt_text
    assert "上一轮回答" in prompt_text
    assert "[memory:101] (preference) 使用中文回答。" in prompt_text
    assert [message.content for message in messages] == [
        "上一轮问题",
        "上一轮回答",
        "如何确认副作用操作？",
        "假的回答",
    ]

def test_chat_workflow_graph_stops_before_model_and_redis_on_memory_error():
    chat_calls = []

    def fake_response(prompt_value):
        chat_calls.append(prompt_value)
        return AIMessage(content="不应生成")

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=FakeRetriever(),
        semantic_memory_service=FailingSemanticMemoryService(),
        chat_model=RunnableLambda(fake_response),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:workflow-memory-error",
            }
        },
    )

    assert result.get("error") == (
        "ConnectionError: long-term memory unavailable"
    )
    assert chat_calls == []
    assert history_store.get("session-a").messages == []

def test_chat_workflow_graph_does_not_save_history_after_model_error():
    def failing_response(prompt_value):
        raise ConnectionError("chat model unavailable")

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(failing_response),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={
            "configurable": {
                "thread_id": "week08:frank:workflow-model-error",
            }
        },
    )

    assert result.get("error") == (
        "ConnectionError: chat model unavailable"
    )
    assert history_store.get("session-a").messages == []
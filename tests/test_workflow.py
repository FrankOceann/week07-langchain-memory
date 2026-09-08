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
from app.conversation import build_conversation_key
from app.tools import InMemoryNotificationTool

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
    history_store.get(
        build_conversation_key("frank", "session-a")
    ).add_messages(
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
    messages = history_store.get(
        build_conversation_key("frank", "session-a")
    ).messages

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


def test_chat_workflow_graph_rewrites_follow_up_before_retrieval():
    def fake_response(prompt_value):
        return AIMessage(content="假的回答")

    rewrite_prompts = []
    decision_prompts = []

    def fake_rewrite(prompt_value):
        rewrite_prompts.append(prompt_value.messages)
        return AIMessage(
            content="为什么执行副作用操作前必须获得用户明确确认？"
        )

    def fake_decider(prompt_value):
        decision_prompts.append(prompt_value.messages)
        return AIMessage(content="REWRITE")

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    history_store.get(
        build_conversation_key("frank", "session-a")
    ).add_messages(
        [
            HumanMessage(content="如何确认副作用操作？"),
            AIMessage(content="执行前必须获得用户明确确认。"),
        ]
    )
    retriever = FakeRetriever()
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=retriever,
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(fake_response),
        query_rewriter=RunnableLambda(fake_rewrite),
        query_rewrite_decider=RunnableLambda(fake_decider),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "那为什么？",
        },
        config={"configurable": {"thread_id": "rewrite-follow-up"}},
    )

    assert result["retrieval_query"] == (
        "为什么执行副作用操作前必须获得用户明确确认？"
    )
    assert retriever.questions == [
        "为什么执行副作用操作前必须获得用户明确确认？"
    ]
    assert result["should_rewrite"] is True
    assert len(decision_prompts) == 1
    assert len(rewrite_prompts) == 1


def test_chat_workflow_graph_skips_rewrite_for_complete_question():
    def fake_response(prompt_value):
        return AIMessage(content="假的回答")

    rewrite_prompts = []

    def fake_rewrite(prompt_value):
        rewrite_prompts.append(prompt_value.messages)
        return AIMessage(content="不应调用改写器")

    def fake_decider(prompt_value):
        return AIMessage(content="KEEP")

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    history_store.get(
        build_conversation_key("frank", "session-a")
    ).add_messages(
        [
            HumanMessage(content="如何确认副作用操作？"),
            AIMessage(content="执行前必须获得用户明确确认。"),
        ]
    )
    retriever = FakeRetriever()
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=retriever,
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(fake_response),
        query_rewriter=RunnableLambda(fake_rewrite),
        query_rewrite_decider=RunnableLambda(fake_decider),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={"configurable": {"thread_id": "keep-complete-query"}},
    )

    assert result["should_rewrite"] is False
    assert result["retrieval_query"] == "如何确认副作用操作？"
    assert retriever.questions == ["如何确认副作用操作？"]
    assert rewrite_prompts == []


def test_chat_workflow_graph_stops_when_rewrite_decision_fails():
    def failing_decider(prompt_value):
        raise ConnectionError("query decision unavailable")

    chat_calls = []
    rewrite_calls = []

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    history_store.get(
        build_conversation_key("frank", "session-a")
    ).add_messages(
        [
            HumanMessage(content="如何确认副作用操作？"),
            AIMessage(content="执行前必须获得用户明确确认。"),
        ]
    )
    retriever = FakeRetriever()
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=retriever,
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: chat_calls.append(prompt_value)
        ),
        query_rewriter=RunnableLambda(
            lambda prompt_value: rewrite_calls.append(prompt_value)
        ),
        query_rewrite_decider=RunnableLambda(failing_decider),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "那为什么？",
        },
        config={"configurable": {"thread_id": "decision-error"}},
    )

    assert result["error"] == "ConnectionError: query decision unavailable"
    assert retriever.questions == []
    assert rewrite_calls == []
    assert chat_calls == []


def test_chat_workflow_graph_interrupts_before_sending_notification():
    notification_tool = InMemoryNotificationTool()
    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: AIMessage(content="已准备发送通知。")
        ),
        notification_tool=notification_tool,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "/notify 项目测试已经完成",
        },
        config={"configurable": {"thread_id": "notify-interrupt"}},
    )

    interrupts = result["__interrupt__"]

    assert interrupts[0].value == {
        "action": "send_notification",
        "message": "项目测试已经完成",
        "impact": "向模拟外部通知服务发送一条消息",
        "cancellable": True,
    }
    assert notification_tool.sent_messages == []


def test_chat_workflow_graph_sends_notification_after_approval():
    notification_tool = InMemoryNotificationTool()
    graph = workflow.build_chat_workflow_graph(
        history_store=RedisHistoryStore(
            fakeredis.FakeRedis(decode_responses=True),
            max_turns=3,
            ttl_seconds=30,
        ),
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: AIMessage(content="已准备发送通知。")
        ),
        notification_tool=notification_tool,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "notify-approve"}}

    graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "/notify 项目测试已经完成",
        },
        config=config,
    )
    result = graph.invoke(Command(resume="approved"), config=config)

    assert notification_tool.sent_messages == ["项目测试已经完成"]
    assert result["action_status"] == "executed"
    assert result["action_result"] == "模拟通知已发送：项目测试已经完成"


def test_chat_workflow_graph_does_not_send_notification_after_rejection():
    notification_tool = InMemoryNotificationTool()
    graph = workflow.build_chat_workflow_graph(
        history_store=RedisHistoryStore(
            fakeredis.FakeRedis(decode_responses=True),
            max_turns=3,
            ttl_seconds=30,
        ),
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: AIMessage(content="已准备发送通知。")
        ),
        notification_tool=notification_tool,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "notify-reject"}}

    graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "/notify 项目测试已经完成",
        },
        config=config,
    )
    result = graph.invoke(Command(resume="rejected"), config=config)

    assert notification_tool.sent_messages == []
    assert result["action_status"] == "rejected"
    assert result["action_result"] == "已取消模拟通知。"


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


def test_chat_workflow_graph_isolates_same_session_id_between_users():
    received_prompts = []

    def fake_response(prompt_value):
        received_prompts.append(prompt_value.messages)
        return AIMessage(content="假的回答")

    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(fake_response),
        checkpointer=InMemorySaver(),
    )

    for user_id, question in [
        ("alice", "Alice 的私密问题"),
        ("bob", "Bob 的独立问题"),
    ]:
        graph.invoke(
            {
                "session_id": "shared-session",
                "user_id": user_id,
                "question": question,
            },
            config={"configurable": {"thread_id": user_id}},
        )

    bob_prompt = "\n".join(
        message.content for message in received_prompts[1]
    )

    assert "Alice 的私密问题" not in bob_prompt
    assert "Bob 的独立问题" in bob_prompt


def test_chat_workflow_graph_stops_before_model_and_redis_on_retrieval_error():
    class FailingRetriever:
        def invoke(self, question: str):
            raise ConnectionError("retriever unavailable")

    chat_calls = []
    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=FailingRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: chat_calls.append(prompt_value)
        ),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={"configurable": {"thread_id": "retrieval-error"}},
    )

    assert result["error"] == "ConnectionError: retriever unavailable"
    assert chat_calls == []
    assert history_store.get("session-a").messages == []


def test_chat_workflow_graph_retries_transient_retrieval_failure_once():
    class TransientFailingRetriever:
        def __init__(self):
            self.questions = []

        def invoke(self, question: str):
            self.questions.append(question)
            if len(self.questions) == 1:
                raise ConnectionError("retriever temporarily unavailable")
            return [
                Document(
                    page_content="副作用操作执行前必须确认。",
                    metadata={"source": "agent_safety.txt#chunk-0"},
                )
            ]

    retriever = TransientFailingRetriever()
    graph = workflow.build_chat_workflow_graph(
        history_store=RedisHistoryStore(
            fakeredis.FakeRedis(decode_responses=True),
            max_turns=3,
            ttl_seconds=30,
        ),
        retriever=retriever,
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: AIMessage(content="第二次检索成功。")
        ),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={"configurable": {"thread_id": "retrieval-retry-success"}},
    )

    assert result["answer"] == "第二次检索成功。"
    assert result["sources"] == ["agent_safety.txt#chunk-0"]
    assert result["error"] is None
    assert retriever.questions == ["如何确认副作用操作？"] * 2


def test_rag_answer_subgraph_generates_answer_from_retrieved_context():
    graph = workflow.build_rag_answer_subgraph(
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: AIMessage(content="子图生成的回答。")
        ),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
            "retrieval_query": "如何确认副作用操作？",
            "history": [],
        }
    )

    assert result["answer"] == "子图生成的回答。"
    assert result["sources"] == [
        "agent_safety.txt#chunk-0",
        "agent_safety.txt#chunk-1",
    ]
    assert result["error"] is None


def test_chat_workflow_graph_returns_fallback_after_retry_is_exhausted():
    class AlwaysFailingRetriever:
        def __init__(self):
            self.questions = []

        def invoke(self, question: str):
            self.questions.append(question)
            raise ConnectionError("retriever unavailable")

    chat_calls = []
    retriever = AlwaysFailingRetriever()
    semantic_memory_service = FakeSemanticMemoryService()
    history_store = RedisHistoryStore(
        fakeredis.FakeRedis(decode_responses=True),
        max_turns=3,
        ttl_seconds=30,
    )
    graph = workflow.build_chat_workflow_graph(
        history_store=history_store,
        retriever=retriever,
        semantic_memory_service=semantic_memory_service,
        chat_model=RunnableLambda(
            lambda prompt_value: chat_calls.append(prompt_value)
        ),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={"configurable": {"thread_id": "retrieval-retry-fallback"}},
    )

    assert result["answer"] == "检索服务暂时不可用，请稍后重试。"
    assert result["sources"] == []
    assert retriever.questions == ["如何确认副作用操作？"] * 2
    assert semantic_memory_service.calls == []
    assert chat_calls == []
    assert history_store.get("session-a").messages == []


def test_chat_workflow_graph_does_not_retry_non_transient_retrieval_error():
    class InvalidRequestRetriever:
        def __init__(self):
            self.questions = []

        def invoke(self, question: str):
            self.questions.append(question)
            raise ValueError("invalid retrieval request")

    retriever = InvalidRequestRetriever()
    graph = workflow.build_chat_workflow_graph(
        history_store=RedisHistoryStore(
            fakeredis.FakeRedis(decode_responses=True),
            max_turns=3,
            ttl_seconds=30,
        ),
        retriever=retriever,
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: AIMessage(content="不应调用模型")
        ),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={"configurable": {"thread_id": "retrieval-non-transient"}},
    )

    assert result["answer"] == "检索服务暂时不可用，请稍后重试。"
    assert retriever.questions == ["如何确认副作用操作？"]


def test_chat_workflow_graph_stops_when_redis_history_load_fails():
    class FailingHistoryStore:
        def get(self, key: str):
            raise ConnectionError("redis unavailable")

    chat_calls = []
    graph = workflow.build_chat_workflow_graph(
        history_store=FailingHistoryStore(),
        retriever=FakeRetriever(),
        semantic_memory_service=FakeSemanticMemoryService(),
        chat_model=RunnableLambda(
            lambda prompt_value: chat_calls.append(prompt_value)
        ),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "session_id": "session-a",
            "user_id": "frank",
            "question": "如何确认副作用操作？",
        },
        config={"configurable": {"thread_id": "history-error"}},
    )

    assert result["error"] == "ConnectionError: redis unavailable"
    assert chat_calls == []

from typing import TypedDict
from app.conversation import build_conversation_key
from app.long_term_memory import render_long_term_memories
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage, HumanMessage



class MinimalWorkflowState(TypedDict, total=False):
    question: str
    completed_questions: list[str]
    answer: str
    error: str | None
    requires_approval: bool
    approval_decision: str


def validate_question(
    state: MinimalWorkflowState,
) -> MinimalWorkflowState:
    question = state["question"].strip()

    if not question:
        return {
            "question": question,
            "error": "问题不能为空。",
        }

    return {
        "question": question,
        "error": None,
    }


def route_after_validation(
    state: MinimalWorkflowState,
) -> str:
    if state["error"] is not None:
        return "error"

    if state.get("requires_approval", False):
        return "approval"

    return "process"


def request_approval(
    state: MinimalWorkflowState,
) -> MinimalWorkflowState:
    decision = interrupt(
        {
            "action": "process_question",
            "question": state["question"],
        }
    )

    return {
        "approval_decision": decision,
    }


def route_after_approval(
    state: MinimalWorkflowState,
) -> str:
    if state["approval_decision"] == "approved":
        return "process"

    return "rejected"


def record_rejection(
    state: MinimalWorkflowState,
) -> MinimalWorkflowState:
    return {
        "error": "人工审批未通过。",
    }


def process_question(
    state: MinimalWorkflowState,
) -> MinimalWorkflowState:
    question = state["question"]

    return {
        "answer": f"已处理：{question}",
        "completed_questions": (
            state.get("completed_questions", []) + [question]
        ),
    }


def build_minimal_graph(checkpointer):
    builder = StateGraph(MinimalWorkflowState)

    builder.add_node("validate_question", validate_question)
    builder.add_node("request_approval", request_approval)
    builder.add_node("record_rejection", record_rejection)
    builder.add_node("process_question", process_question)

    builder.add_edge(START, "validate_question")
    builder.add_conditional_edges(
        "validate_question",
        route_after_validation,
        {
            "error": END,
            "approval": "request_approval",
            "process": "process_question",
        },
    )
    builder.add_conditional_edges(
        "request_approval",
        route_after_approval,
        {
            "process": "process_question",
            "rejected": "record_rejection",
        },
    )
    builder.add_edge("record_rejection", END)
    builder.add_edge("process_question", END)

    return builder.compile(checkpointer=checkpointer)

class RagRetrievalState(TypedDict, total=False):
    question: str
    context: str
    sources: list[str]
    error: str | None


def build_rag_retrieval_graph(retriever, checkpointer):
    def retrieve_rag(
        state: RagRetrievalState,
    ) -> RagRetrievalState:
        try:
            documents = retriever.invoke(state["question"])
        except Exception as error:
            return {
                "error": (
                    f"{type(error).__name__}: {error}"
                )
            }

        sources = [
            document.metadata["source"]
            for document in documents
        ]
        context = "\n\n".join(
            f"[{document.metadata['source']}]\n"
            f"{document.page_content}"
            for document in documents
        )

        return {
            "context": context,
            "sources": sources,
            "error": None,
        }

    builder = StateGraph(RagRetrievalState)

    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_edge(START, "retrieve_rag")
    builder.add_edge("retrieve_rag", END)

    return builder.compile(checkpointer=checkpointer)

class RagMemoryState(TypedDict, total=False):
    question: str
    user_id: str
    context: str
    sources: list[str]
    long_term_memory_context: str
    error: str | None


def build_rag_memory_graph(
    retriever,
    semantic_memory_service,
    checkpointer,
):
    def retrieve_rag(
        state: RagMemoryState,
    ) -> RagMemoryState:
        try:
            documents = retriever.invoke(state["question"])
        except Exception as error:
            return {
                "error": (
                    f"{type(error).__name__}: {error}"
                )
            }

        sources = [
            document.metadata["source"]
            for document in documents
        ]
        context = "\n\n".join(
            f"[{document.metadata['source']}]\n"
            f"{document.page_content}"
            for document in documents
        )

        return {
            "context": context,
            "sources": sources,
            "error": None,
        }

    def route_after_retrieval(
        state: RagMemoryState,
    ) -> str:
        if state["error"] is not None:
            return "error"

        return "memory"

    def load_long_term_memory(
        state: RagMemoryState,
    ) -> RagMemoryState:
        try:
            memories = semantic_memory_service.search_active(
                user_id=state["user_id"],
                question=state["question"],
            )
        except Exception as error:
            return {
                "error": (
                    f"{type(error).__name__}: {error}"
                )
            }

        return {
            "long_term_memory_context": (
                render_long_term_memories(memories)
            ),
            "error": None,
        }

    builder = StateGraph(RagMemoryState)

    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node(
        "load_long_term_memory",
        load_long_term_memory,
    )

    builder.add_edge(START, "retrieve_rag")
    builder.add_conditional_edges(
        "retrieve_rag",
        route_after_retrieval,
        {
            "error": END,
            "memory": "load_long_term_memory",
        },
    )
    builder.add_edge("load_long_term_memory", END)

    return builder.compile(checkpointer=checkpointer)

class RagMemoryAnswerState(TypedDict, total=False):
    question: str
    user_id: str
    context: str
    sources: list[str]
    long_term_memory_context: str
    answer: str
    error: str | None


def build_rag_memory_answer_graph(
    retriever,
    semantic_memory_service,
    chat_model,
    checkpointer,
):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只依据本轮检索资料回答；资料未覆盖时回答“资料不足”。"
                "长期记忆仅用于个性化参考，不能作为新事实来源。",
            ),
            (
                "human",
                "已确认长期记忆（仅用于个性化参考，不能覆盖本轮检索资料）："
                "\n{long_term_memories}\n\n"
                "本轮检索资料：\n{context}\n\n当前问题：{question}",
            ),
        ]
    )

    def retrieve_rag(
        state: RagMemoryAnswerState,
    ) -> RagMemoryAnswerState:
        try:
            documents = retriever.invoke(state["question"])
        except Exception as error:
            return {
                "error": (
                    f"{type(error).__name__}: {error}"
                )
            }

        sources = [
            document.metadata["source"]
            for document in documents
        ]
        context = "\n\n".join(
            f"[{document.metadata['source']}]\n"
            f"{document.page_content}"
            for document in documents
        )

        return {
            "context": context,
            "sources": sources,
            "error": None,
        }

    def route_after_retrieval(
        state: RagMemoryAnswerState,
    ) -> str:
        if state["error"] is not None:
            return "error"

        return "memory"

    def load_long_term_memory(
        state: RagMemoryAnswerState,
    ) -> RagMemoryAnswerState:
        try:
            memories = semantic_memory_service.search_active(
                user_id=state["user_id"],
                question=state["question"],
            )
        except Exception as error:
            return {
                "error": (
                    f"{type(error).__name__}: {error}"
                )
            }

        return {
            "long_term_memory_context": (
                render_long_term_memories(memories)
            ),
            "error": None,
        }

    def route_after_memory(
        state: RagMemoryAnswerState,
    ) -> str:
        if state["error"] is not None:
            return "error"

        return "answer"

    def generate_answer(
        state: RagMemoryAnswerState,
    ) -> RagMemoryAnswerState:
        try:
            response = (prompt | chat_model).invoke(
                {
                    "question": state["question"],
                    "context": state["context"],
                    "long_term_memories": (
                        state["long_term_memory_context"]
                    ),
                }
            )
        except Exception as error:
            return {
                "error": (
                    f"{type(error).__name__}: {error}"
                )
            }

        return {
            "answer": response.content,
            "error": None,
        }

    builder = StateGraph(RagMemoryAnswerState)

    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node(
        "load_long_term_memory",
        load_long_term_memory,
    )
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "retrieve_rag")
    builder.add_conditional_edges(
        "retrieve_rag",
        route_after_retrieval,
        {
            "error": END,
            "memory": "load_long_term_memory",
        },
    )
    builder.add_conditional_edges(
        "load_long_term_memory",
        route_after_memory,
        {
            "error": END,
            "answer": "generate_answer",
        },
    )
    builder.add_edge("generate_answer", END)

    return builder.compile(checkpointer=checkpointer)

class ChatHistoryState(TypedDict, total=False):
    session_id: str
    question: str
    answer: str
    history: list
    error: str | None


def build_chat_history_graph(history_store, checkpointer):
    def load_short_history(
        state: ChatHistoryState,
    ) -> ChatHistoryState:
        history = history_store.get(state["session_id"])

        return {
            "history": history.messages,
        }

    def route_after_history_load(
        state: ChatHistoryState,
    ) -> str:
        if (
            state.get("error") is None
            and "question" in state
            and "answer" in state
        ):
            return "save"

        return "end"

    def save_short_history(
        state: ChatHistoryState,
    ) -> ChatHistoryState:
        history = history_store.get(state["session_id"])
        history.add_messages(
            [
                HumanMessage(content=state["question"]),
                AIMessage(content=state["answer"]),
            ]
        )

        return {}

    builder = StateGraph(ChatHistoryState)

    builder.add_node("load_short_history", load_short_history)
    builder.add_node("save_short_history", save_short_history)

    builder.add_edge(START, "load_short_history")
    builder.add_conditional_edges(
        "load_short_history",
        route_after_history_load,
        {
            "save": "save_short_history",
            "end": END,
        },
    )
    builder.add_edge("save_short_history", END)

    return builder.compile(checkpointer=checkpointer)

class ChatWorkflowState(TypedDict, total=False):
    session_id: str
    user_id: str
    question: str
    history: list
    should_rewrite: bool
    retrieval_query: str
    retrieval_attempts: int
    retry_retrieval: bool
    context: str
    sources: list[str]
    long_term_memory_context: str
    answer: str
    pending_action: dict | None
    approval_decision: str
    action_status: str
    action_result: str
    error: str | None


def build_rag_answer_subgraph(
    retriever,
    semantic_memory_service,
    chat_model,
):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只依据本轮检索资料回答；资料未覆盖时回答“资料不足”。"
                "历史仅用于理解指代，不能作为新事实来源。"
                "长期记忆仅用于个性化参考，不能作为新事实来源。",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "已确认长期记忆（仅用于个性化参考，不能覆盖本轮检索资料）："
                "\n{long_term_memories}\n\n"
                "本轮检索资料：\n{context}\n\n当前问题：{question}",
            ),
        ]
    )

    def retrieve_rag(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        try:
            documents = retriever.invoke(state["retrieval_query"])
        except Exception as error:
            retrieval_attempts = state.get("retrieval_attempts", 0) + 1
            return {
                "error": f"{type(error).__name__}: {error}",
                "retrieval_attempts": retrieval_attempts,
                "retry_retrieval": (
                    isinstance(error, ConnectionError)
                    and retrieval_attempts <= 1
                ),
            }

        sources = [document.metadata["source"] for document in documents]
        context = "\n\n".join(
            f"[{document.metadata['source']}]\n{document.page_content}"
            for document in documents
        )
        return {
            "context": context,
            "sources": sources,
            "retry_retrieval": False,
            "error": None,
        }

    def route_after_retrieval(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            if state.get("retry_retrieval", False):
                return "retry"
            return "fallback"
        return "memory"

    def generate_retrieval_fallback(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        return {
            "answer": "检索服务暂时不可用，请稍后重试。",
            "sources": [],
        }

    def load_long_term_memory(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        try:
            memories = semantic_memory_service.search_active(
                user_id=state["user_id"],
                question=state["question"],
            )
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

        return {
            "long_term_memory_context": render_long_term_memories(memories),
            "error": None,
        }

    def route_after_memory(
        state: ChatWorkflowState,
    ) -> str:
        return "end" if state["error"] is not None else "answer"

    def generate_answer(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        try:
            response = (prompt | chat_model).invoke(
                {
                    "history": state["history"],
                    "question": state["question"],
                    "context": state["context"],
                    "long_term_memories": state["long_term_memory_context"],
                }
            )
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

        return {"answer": response.content, "error": None}

    builder = StateGraph(ChatWorkflowState)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node(
        "generate_retrieval_fallback",
        generate_retrieval_fallback,
    )
    builder.add_node("load_long_term_memory", load_long_term_memory)
    builder.add_node("generate_answer", generate_answer)
    builder.add_edge(START, "retrieve_rag")
    builder.add_conditional_edges(
        "retrieve_rag",
        route_after_retrieval,
        {
            "retry": "retrieve_rag",
            "fallback": "generate_retrieval_fallback",
            "memory": "load_long_term_memory",
        },
    )
    builder.add_edge("generate_retrieval_fallback", END)
    builder.add_conditional_edges(
        "load_long_term_memory",
        route_after_memory,
        {"end": END, "answer": "generate_answer"},
    )
    builder.add_edge("generate_answer", END)
    return builder.compile()


def build_chat_workflow_inspector(checkpointer):
    builder = StateGraph(ChatWorkflowState)
    return builder.compile(checkpointer=checkpointer)


def build_chat_workflow_graph(
    history_store,
    retriever,
    semantic_memory_service,
    chat_model,
    checkpointer,
    query_rewriter=None,
    query_rewrite_decider=None,
    notification_tool=None,
):
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "结合历史对话，将当前问题改写成适合知识库检索的完整问题。"
                "不要添加历史中不存在的事实；只输出改写后的问题。",
            ),
            MessagesPlaceholder("history"),
            ("human", "当前问题：{question}"),
        ]
    )
    rewrite_decision_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "判断当前问题是否必须结合历史对话改写后才能检索知识库。"
                "问题完整清晰时输出 KEEP；问题含有指代或省略时输出 REWRITE。"
                "只输出 KEEP 或 REWRITE。",
            ),
            MessagesPlaceholder("history"),
            ("human", "当前问题：{question}"),
        ]
    )

    def load_short_history(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        try:
            history = history_store.get(
                build_conversation_key(
                    state["user_id"],
                    state["session_id"],
                )
            )
        except Exception as error:
            return {
                "error": f"{type(error).__name__}: {error}"
            }

        return {
            "history": history.messages,
            "error": None,
        }

    def route_after_history_load(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        return "decision"

    def decide_query_rewrite(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        if not state["history"] or query_rewriter is None:
            return {
                "should_rewrite": False,
                "retrieval_query": state["question"],
                "error": None,
            }

        if query_rewrite_decider is None:
            return {
                "should_rewrite": True,
                "error": None,
            }

        try:
            response = (rewrite_decision_prompt | query_rewrite_decider).invoke(
                {
                    "history": state["history"],
                    "question": state["question"],
                }
            )
            decision = response.content.strip().upper()
            if decision not in {"KEEP", "REWRITE"}:
                raise ValueError(
                    "查询改写判断器必须返回 KEEP 或 REWRITE。"
                )
        except Exception as error:
            return {
                "error": f"{type(error).__name__}: {error}"
            }

        return {
            "should_rewrite": decision == "REWRITE",
            "retrieval_query": (
                state["question"] if decision == "KEEP" else ""
            ),
            "error": None,
        }

    def route_after_query_decision(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        if state["should_rewrite"]:
            return "rewrite"

        return "rag_answer"

    def rewrite_retrieval_query(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        if not state["history"] or query_rewriter is None:
            return {
                "retrieval_query": state["question"],
                "error": None,
            }

        try:
            response = (rewrite_prompt | query_rewriter).invoke(
                {
                    "history": state["history"],
                    "question": state["question"],
                }
            )
            retrieval_query = response.content.strip()
        except Exception as error:
            return {
                "error": f"{type(error).__name__}: {error}"
            }

        return {
            "retrieval_query": retrieval_query or state["question"],
            "error": None,
        }

    def route_after_query_rewrite(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        return "rag_answer"

    def route_after_rag_answer_subgraph(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        return "plan_action"

    def plan_notification_action(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        notification_prefix = "/notify "
        question = state["question"].strip()

        if not question.startswith(notification_prefix):
            return {
                "pending_action": None,
                "action_status": "not_requested",
                "error": None,
            }

        message = question.removeprefix(notification_prefix).strip()
        if not message:
            return {"error": "通知内容不能为空。"}

        if notification_tool is None:
            return {"error": "未配置模拟通知工具。"}

        return {
            "pending_action": {
                "action": "send_notification",
                "message": message,
                "impact": "向模拟外部通知服务发送一条消息",
                "cancellable": True,
            },
            "action_status": "pending_approval",
            "error": None,
        }

    def route_after_action_plan(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        if state["pending_action"] is None:
            return "save"

        return "approval"

    def request_notification_approval(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        decision = interrupt(state["pending_action"])

        return {"approval_decision": decision}

    def route_after_action_approval(
        state: ChatWorkflowState,
    ) -> str:
        if state["approval_decision"] == "approved":
            return "execute"

        return "reject"

    def execute_notification_action(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        try:
            action_result = notification_tool.send(
                state["pending_action"]["message"]
            )
        except Exception as error:
            return {
                "error": f"{type(error).__name__}: {error}"
            }

        return {
            "action_status": "executed",
            "action_result": action_result,
            "answer": (
                f"{state['answer']}\n\n操作结果：{action_result}"
            ),
            "error": None,
        }

    def route_after_action_execution(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        return "save"

    def record_notification_rejection(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        action_result = "已取消模拟通知。"

        return {
            "action_status": "rejected",
            "action_result": action_result,
            "answer": (
                f"{state['answer']}\n\n操作结果：{action_result}"
            ),
            "error": None,
        }

    def save_short_history(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        try:
            history = history_store.get(
                build_conversation_key(
                    state["user_id"],
                    state["session_id"],
                )
            )
            history.add_messages(
                [
                    HumanMessage(content=state["question"]),
                    AIMessage(content=state["answer"]),
                ]
            )
        except Exception as error:
            return {
                "error": f"{type(error).__name__}: {error}"
            }

        return {}

    rag_answer_subgraph = build_rag_answer_subgraph(
        retriever=retriever,
        semantic_memory_service=semantic_memory_service,
        chat_model=chat_model,
    )
    builder = StateGraph(ChatWorkflowState)

    builder.add_node("load_short_history", load_short_history)
    builder.add_node("decide_query_rewrite", decide_query_rewrite)
    builder.add_node("rewrite_retrieval_query", rewrite_retrieval_query)
    builder.add_node("rag_answer_subgraph", rag_answer_subgraph)
    builder.add_node("plan_notification_action", plan_notification_action)
    builder.add_node(
        "request_notification_approval",
        request_notification_approval,
    )
    builder.add_node(
        "execute_notification_action",
        execute_notification_action,
    )
    builder.add_node(
        "record_notification_rejection",
        record_notification_rejection,
    )
    builder.add_node("save_short_history", save_short_history)

    builder.add_edge(START, "load_short_history")
    builder.add_conditional_edges(
        "load_short_history",
        route_after_history_load,
        {
            "end": END,
            "decision": "decide_query_rewrite",
        },
    )
    builder.add_conditional_edges(
        "decide_query_rewrite",
        route_after_query_decision,
        {
            "end": END,
            "rewrite": "rewrite_retrieval_query",
            "rag_answer": "rag_answer_subgraph",
        },
    )
    builder.add_conditional_edges(
        "rewrite_retrieval_query",
        route_after_query_rewrite,
        {
            "end": END,
            "rag_answer": "rag_answer_subgraph",
        },
    )
    builder.add_conditional_edges(
        "rag_answer_subgraph",
        route_after_rag_answer_subgraph,
        {
            "end": END,
            "plan_action": "plan_notification_action",
        },
    )
    builder.add_conditional_edges(
        "plan_notification_action",
        route_after_action_plan,
        {
            "end": END,
            "approval": "request_notification_approval",
            "save": "save_short_history",
        },
    )
    builder.add_conditional_edges(
        "request_notification_approval",
        route_after_action_approval,
        {
            "execute": "execute_notification_action",
            "reject": "record_notification_rejection",
        },
    )
    builder.add_conditional_edges(
        "execute_notification_action",
        route_after_action_execution,
        {
            "end": END,
            "save": "save_short_history",
        },
    )
    builder.add_edge("record_notification_rejection", "save_short_history")
    builder.add_edge("save_short_history", END)

    return builder.compile(checkpointer=checkpointer)

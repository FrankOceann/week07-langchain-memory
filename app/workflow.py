from typing import TypedDict
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
    context: str
    sources: list[str]
    long_term_memory_context: str
    answer: str
    error: str | None


def build_chat_workflow_graph(
    history_store,
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

    def load_short_history(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        history = history_store.get(state["session_id"])

        return {
            "history": history.messages,
        }

    def retrieve_rag(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
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
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        return "memory"

    def load_long_term_memory(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
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
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        return "answer"

    def generate_answer(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        try:
            response = (prompt | chat_model).invoke(
                {
                    "history": state["history"],
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

    def route_after_answer(
        state: ChatWorkflowState,
    ) -> str:
        if state["error"] is not None:
            return "end"

        return "save"

    def save_short_history(
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        history = history_store.get(state["session_id"])
        history.add_messages(
            [
                HumanMessage(content=state["question"]),
                AIMessage(content=state["answer"]),
            ]
        )

        return {}

    builder = StateGraph(ChatWorkflowState)

    builder.add_node("load_short_history", load_short_history)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node(
        "load_long_term_memory",
        load_long_term_memory,
    )
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("save_short_history", save_short_history)

    builder.add_edge(START, "load_short_history")
    builder.add_edge("load_short_history", "retrieve_rag")
    builder.add_conditional_edges(
        "retrieve_rag",
        route_after_retrieval,
        {
            "end": END,
            "memory": "load_long_term_memory",
        },
    )
    builder.add_conditional_edges(
        "load_long_term_memory",
        route_after_memory,
        {
            "end": END,
            "answer": "generate_answer",
        },
    )
    builder.add_conditional_edges(
        "generate_answer",
        route_after_answer,
        {
            "end": END,
            "save": "save_short_history",
        },
    )
    builder.add_edge("save_short_history", END)

    return builder.compile(checkpointer=checkpointer)
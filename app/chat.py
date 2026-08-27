from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.embeddings import DEFAULT_BASE_URL

def build_chat_model(api_key: str | None = None) -> ChatOpenAI:
    load_dotenv()

    resolved_key = (
        api_key
        if api_key is not None
        else os.getenv("DASHSCOPE_API_KEY", "")
    )

    if not resolved_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，无法创建聊天模型。")

    return ChatOpenAI(
        model="qwen-plus",
        api_key=resolved_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        temperature=0,
    )

def build_conversation_runnable(chat_model, history_store):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只依据本轮检索资料回答；资料未覆盖时回答“资料不足”。"
                "历史仅用于理解指代，不能作为新事实来源。",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "本轮检索资料：\n{context}\n\n当前问题：{question}",
            ),
        ]
    )

    return RunnableWithMessageHistory(
        prompt | chat_model,
        history_store.get,
        input_messages_key="question",
        history_messages_key="history",
    )


def ask_question(
    question: str,
    session_id: str,
    retriever,
    conversation_runnable,
) -> tuple[str, list[str]]:
    documents = retriever.invoke(question)
    sources = [document.metadata["source"] for document in documents]
    context = "\n\n".join(
        f"[{document.metadata['source']}]\n{document.page_content}"
        for document in documents
    )

    response = conversation_runnable.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}},
    )

    return response.content, sources
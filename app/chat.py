import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.conversation import build_workflow_thread_id
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


def ask_question_with_workflow(
    question: str,
    session_id: str,
    user_id: str,
    workflow_graph,
) -> tuple[str, list[str]]:
    result = workflow_graph.invoke(
        {
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
        },
        config={
            "configurable": {
                "thread_id": build_workflow_thread_id(
                    user_id,
                    session_id,
                ),
            }
        },
    )

    if error := result.get("error"):
        raise RuntimeError(error)

    return result["answer"], result["sources"]

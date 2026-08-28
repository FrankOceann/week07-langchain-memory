import sys
from pathlib import Path

from app.chat import (
    ask_question,
    build_chat_model,
    build_conversation_runnable,
)
from app.embeddings import DashScopeEmbeddings
from app.memory import RedisHistoryStore, build_redis_client
from app.retriever import build_retriever


DATA_DIRECTORY = Path(__file__).parent / "data"
DEFAULT_SESSION_ID = "demo-session"


def validate_question(question: str) -> str:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("问题不能为空。")

    return normalized_question


def main() -> int:
    session_id = (
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SESSION_ID
    )

    try:
        retriever = build_retriever(
            DATA_DIRECTORY,
            DashScopeEmbeddings(),
        )
        history_store = RedisHistoryStore(
            build_redis_client(),
            max_turns=3,
            ttl_seconds=1800,
        )
        conversation_runnable = build_conversation_runnable(
            build_chat_model(),
            history_store,
        )
    except ValueError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    print(f"当前会话：{session_id}")
    print("输入 exit、quit 或 退出，结束对话。")

    while True:
        try:
            raw_question = input("你：")
        except EOFError:
            return 0

        if raw_question.strip().lower() in {"exit", "quit", "退出"}:
            return 0

        try:
            question = validate_question(raw_question)
        except ValueError as error:
            print(f"错误：{error}")
            continue

        answer, sources = ask_question(
            question,
            session_id=session_id,
            retriever=retriever,
            conversation_runnable=conversation_runnable,
        )

        print(f"助手：{answer}")

        for source in sources:
            print(f"=== {source} ===")


if __name__ == "__main__":
    raise SystemExit(main())
import argparse
import sys
from pathlib import Path
from app.memory_sync import MemorySyncService
from app.milvus_memory import MilvusMemoryVectorIndex, build_milvus_client
from sqlalchemy.exc import SQLAlchemyError
from app.semantic_memory import SemanticLongTermMemoryService
from app.chat import (
    ask_question,
    build_chat_model,
    build_conversation_runnable,
)
from app.database import build_session_factory
from app.embeddings import DashScopeEmbeddings
from app.long_term_memory import (
    ALLOWED_CATEGORIES,
    SQLAlchemyLongTermMemoryRepository,
)
from app.memory import RedisHistoryStore, build_redis_client
from app.retriever import build_retriever


DATA_DIRECTORY = Path(__file__).parent / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    chat = commands.add_parser("chat")
    chat.add_argument("--session-id", required=True)
    chat.add_argument("--user-id", required=True)

    memory = commands.add_parser("memory")
    memory_commands = memory.add_subparsers(
        dest="memory_command",
        required=True,
    )

    add = memory_commands.add_parser("add")
    add.add_argument("--user-id", required=True)
    add.add_argument(
        "--category",
        choices=sorted(ALLOWED_CATEGORIES),
        required=True,
    )
    add.add_argument("--content", required=True)

    list_command = memory_commands.add_parser("list")
    list_command.add_argument("--user-id", required=True)
    list_command.add_argument(
        "--category",
        choices=sorted(ALLOWED_CATEGORIES),
    )
    list_command.add_argument("--limit", type=int, default=5)

    deactivate = memory_commands.add_parser("deactivate")
    deactivate.add_argument("--memory-id", type=int, required=True)

    return parser


def validate_question(question: str) -> str:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("问题不能为空。")

    return normalized_question


def run_memory_command(
    args,
    repository: SQLAlchemyLongTermMemoryRepository,
    memory_sync_service,
) -> int:
    if args.memory_command == "add":
        memory = repository.add(
            user_id=args.user_id,
            category=args.category,
            content=args.content,
        )
        try:
            memory_sync_service.sync(memory)
        except Exception:
            print(
                "长期记忆已写入 MySQL，但 Milvus 同步失败。"
                "请稍后重试同步。",
                file=sys.stderr,
            )
            return 2

        print(f"已新增长期记忆：{memory.id}")
        return 0

    if args.memory_command == "list":
        memories = repository.list_active(
            user_id=args.user_id,
            category=args.category,
            limit=args.limit,
        )

        if not memories:
            print("无有效长期记忆。")
            return 0

        for memory in memories:
            print(f"[{memory.id}] {memory.category}: {memory.content}")

        return 0

    if args.memory_command == "deactivate":
        deactivated = repository.deactivate(args.memory_id)

        if not deactivated:
            print(
                f"未找到有效长期记忆：{args.memory_id}",
                file=sys.stderr,
            )
            return 1

        print(f"已停用长期记忆：{args.memory_id}")
        return 0

    raise ValueError("未知的 memory 子命令。")


def run_chat_command(
    args,
    repository: SQLAlchemyLongTermMemoryRepository,
) -> int:
    embeddings = DashScopeEmbeddings()

    retriever = build_retriever(
        DATA_DIRECTORY,
        embeddings,
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

    vector_index = MilvusMemoryVectorIndex(
        client=build_milvus_client(),
        collection_name="long_term_memory_vectors",
    )
    semantic_memory_service = SemanticLongTermMemoryService(
        embeddings=embeddings,
        vector_index=vector_index,
        long_term_memory_repository=repository,
    )

    print(f"当前会话：{args.session_id}")
    print(f"当前用户：{args.user_id}")
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
            session_id=args.session_id,
            user_id=args.user_id,
            retriever=retriever,
            conversation_runnable=conversation_runnable,
            semantic_memory_service=semantic_memory_service,
        )

        print(f"助手：{answer}")

        for source in sources:
            print(f"=== {source} ===")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        repository = SQLAlchemyLongTermMemoryRepository(
            build_session_factory()
        )

        if args.command == "memory":
            memory_sync_service = None

            if args.memory_command == "add":
                vector_index = MilvusMemoryVectorIndex(
                    client=build_milvus_client(),
                    collection_name="long_term_memory_vectors",
                )
                memory_sync_service = MemorySyncService(
                    embeddings=DashScopeEmbeddings(),
                    vector_index=vector_index,
                )

            return run_memory_command(
                args,
                repository,
                memory_sync_service,
            )

        if args.command == "chat":
            return run_chat_command(args, repository)

        raise ValueError("未知命令。")
    except (ValueError, SQLAlchemyError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
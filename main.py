import sys
from pathlib import Path

from app.embeddings import DashScopeEmbeddings
from app.retriever import build_retriever


DATA_DIRECTORY = Path(__file__).parent / "data"


def validate_question(question: str) -> str:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("问题不能为空。")

    return normalized_question


def main() -> int:
    try:
        question = validate_question(" ".join(sys.argv[1:]))
        retriever = build_retriever(DATA_DIRECTORY, DashScopeEmbeddings())

        for document in retriever.invoke(question):
            print(f"=== {document.metadata['source']} ===")
            print(document.page_content)
            print()

        return 0
    except ValueError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
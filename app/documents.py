from pathlib import Path

from langchain_core.documents import Document


CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def load_documents(data_directory: Path) -> list[Document]:
    documents: list[Document] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for file_path in sorted(data_directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        for chunk_index, start in enumerate(range(0, len(text), step)):
            chunk = text[start : start + CHUNK_SIZE]

            if chunk:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": f"{file_path.name}#chunk-{chunk_index}"},
                    )
                )

    return documents
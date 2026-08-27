from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.documents import load_documents


def build_retriever(
    data_directory: Path,
    embeddings: Embeddings,
    top_k: int = 3,
):
    documents = load_documents(data_directory)

    if not documents:
        raise ValueError("没有可索引资料。")

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    return vector_store.as_retriever(search_kwargs={"k": top_k})


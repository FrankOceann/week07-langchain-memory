from app.documents import load_documents


def test_load_documents_preserves_chunk_source_and_overlap(tmp_path):
    text = "A" * 400 + "B" * 100
    (tmp_path / "guide.txt").write_text(text, encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.metadata["source"] for document in documents] == [
        "guide.txt#chunk-0",
        "guide.txt#chunk-1",
    ]
    assert documents[0].page_content == "A" * 400
    assert documents[1].page_content == "A" * 50 + "B" * 100

def test_load_documents_returns_empty_list_for_empty_directory(tmp_path):
    assert load_documents(tmp_path) == []

from app.embeddings import DashScopeEmbeddings


class FakeEmbeddingsClient:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, input))
        rows = [
            type("Row", (), {"embedding": [float(index), 1.0]})()
            for index, _ in enumerate(input)
        ]
        return type("Response", (), {"data": rows})()


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsClient()


def test_dashscope_embeddings_implements_langchain_methods():
    client = FakeClient()
    embeddings = DashScopeEmbeddings(client=client, api_key="test-key")

    assert embeddings.embed_documents(["第一段", "第二段"]) == [
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    assert embeddings.embed_query("问题") == [0.0, 1.0]
    assert client.embeddings.calls == [
        ("text-embedding-v4", ["第一段", "第二段"]),
        ("text-embedding-v4", ["问题"]),
    ]

import pytest


def test_dashscope_embeddings_rejects_missing_api_key():
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        DashScopeEmbeddings(api_key="")

from langchain_core.embeddings import Embeddings

from app.retriever import build_retriever


class DeterministicEmbeddings(Embeddings):
    vectors = {
        "Python 文件安全": [1.0, 0.0],
        "Agent 权限确认": [0.0, 1.0],
        "RAG 文档切分": [0.8, 0.2],
        "如何确认副作用？": [0.0, 1.0],
    }

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        return self.vectors[text]


def test_build_retriever_returns_relevant_documents_with_source(tmp_path):
    (tmp_path / "python.txt").write_text("Python 文件安全", encoding="utf-8")
    (tmp_path / "agent.txt").write_text("Agent 权限确认", encoding="utf-8")
    (tmp_path / "rag.txt").write_text("RAG 文档切分", encoding="utf-8")

    retriever = build_retriever(tmp_path, DeterministicEmbeddings())
    results = retriever.invoke("如何确认副作用？")

    assert [document.metadata["source"] for document in results] == [
        "agent.txt#chunk-0",
        "rag.txt#chunk-0",
        "python.txt#chunk-0",
    ]
    assert [document.page_content for document in results] == [
        "Agent 权限确认",
        "RAG 文档切分",
        "Python 文件安全",
    ]

def test_build_retriever_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="没有可索引资料"):
        build_retriever(tmp_path, DeterministicEmbeddings())

from main import validate_question


def test_validate_question_rejects_blank_input():
    with pytest.raises(ValueError, match="问题不能为空"):
        validate_question("  ")


def test_validate_question_strips_whitespace():
    assert validate_question("  如何确认副作用？  ") == "如何确认副作用？"
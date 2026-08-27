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
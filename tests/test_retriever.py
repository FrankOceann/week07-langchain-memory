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
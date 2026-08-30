from types import SimpleNamespace

from app.semantic_memory import SemanticLongTermMemoryService


class FakeEmbeddings:
    def __init__(self):
        self.questions = []

    def embed_query(self, text: str) -> list[float]:
        self.questions.append(text)
        return [0.1, 0.2, 0.3]


class FakeVectorIndex:
    def __init__(self):
        self.search_calls = []

    def search(
        self,
        user_id: str,
        vector: list[float],
        limit: int,
    ) -> list[int]:
        self.search_calls.append((user_id, vector, limit))
        return [12, 10, 99]


class FakeRepository:
    def __init__(self):
        self.calls = []

    def list_active_by_ids(
        self,
        user_id: str,
        memory_ids: list[int],
    ):
        self.calls.append((user_id, memory_ids))
        return [
            SimpleNamespace(id=10, content="使用中文回答。"),
            SimpleNamespace(id=12, content="分步骤说明。"),
        ]


def test_semantic_service_preserves_milvus_rank_order():
    embeddings = FakeEmbeddings()
    vector_index = FakeVectorIndex()
    repository = FakeRepository()

    service = SemanticLongTermMemoryService(
        embeddings=embeddings,
        vector_index=vector_index,
        long_term_memory_repository=repository,
    )

    memories = service.search_active(
        user_id="frank",
        question="请按我平时的习惯给出清晰说明。",
        limit=3,
    )

    assert [memory.id for memory in memories] == [12, 10]
    assert embeddings.questions == ["请按我平时的习惯给出清晰说明。"]
    assert vector_index.search_calls == [
        ("frank", [0.1, 0.2, 0.3], 3),
    ]
    assert repository.calls == [
        ("frank", [12, 10, 99]),
    ]
from types import SimpleNamespace

from app.memory_sync import MemorySyncService


class FakeEmbeddings:
    def __init__(self, vector):
        self.vector = vector
        self.questions = []

    def embed_query(self, question):
        self.questions.append(question)
        return self.vector


class RecordingVectorIndex:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, memory_id, user_id, vector):
        self.upsert_calls.append(
            {
                "memory_id": memory_id,
                "user_id": user_id,
                "vector": vector,
            }
        )


def test_sync_embeds_mysql_memory_and_upserts_it_to_milvus():
    vector = [0.1] * 1024
    embeddings = FakeEmbeddings(vector)
    vector_index = RecordingVectorIndex()
    service = MemorySyncService(
        embeddings=embeddings,
        vector_index=vector_index,
    )
    memory = SimpleNamespace(
        id=101,
        user_id="frank",
        content="回答时优先使用中文",
    )

    service.sync(memory)

    assert embeddings.questions == ["回答时优先使用中文"]
    assert vector_index.upsert_calls == [
        {
            "memory_id": 101,
            "user_id": "frank",
            "vector": vector,
        }
    ]
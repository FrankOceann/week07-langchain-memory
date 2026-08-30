import app.milvus_memory as milvus_memory
import pytest
from pymilvus import DataType

from app.milvus_memory import (
    MilvusMemoryVectorIndex,
    build_milvus_client,
    get_milvus_uri,
)


def test_get_milvus_uri_rejects_missing_uri(monkeypatch):
    monkeypatch.delenv("MILVUS_URI", raising=False)

    with pytest.raises(ValueError, match="MILVUS_URI"):
        get_milvus_uri(uri="")

class FakeSearchHit:
    def __init__(self, memory_id):
        self.id = memory_id

class FakeMilvusClient:
    def __init__(self):
        self.upsert_calls = []
        self.search_calls = []
        self.has_collection_calls = []

    def has_collection(self, collection_name: str) -> bool:
        self.has_collection_calls.append(collection_name)
        return True

    def upsert(self, collection_name: str, data: list[dict]):
        self.upsert_calls.append((collection_name, data))

    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        filter: str,
        limit: int,
    ):
        self.search_calls.append(
            (collection_name, data, filter, limit)
        )
        return [[FakeSearchHit(12), FakeSearchHit(10)]]


def test_vector_index_upserts_memory_id_as_primary_key():
    client = FakeMilvusClient()
    vector = [0.0] * 1024

    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    vector_index.upsert(
        memory_id=101,
        user_id="frank",
        vector=vector,
    )

    assert client.upsert_calls == [
        (
            "long_term_memory_vectors",
            [
                {
                    "memory_id": 101,
                    "user_id": "frank",
                    "embedding": vector,
                }
            ],
        )
    ]

def test_vector_index_searches_only_current_user_ids():
    client = FakeMilvusClient()
    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    memory_ids = vector_index.search(
        user_id="frank",
        vector=[0.1] * 1024,
        limit=2,
    )

    assert memory_ids == [12, 10]
    assert client.search_calls == [
        (
            "long_term_memory_vectors",
            [[0.1] * 1024],
            'user_id == "frank"',
            2,
        )
    ]

def test_vector_index_rejects_wrong_vector_dimension():
    client = FakeMilvusClient()
    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    with pytest.raises(ValueError, match="1024"):
        vector_index.upsert(
            memory_id=101,
            user_id="frank",
            vector=[0.1, 0.2],
        )

    assert client.upsert_calls == []

def test_vector_index_rejects_wrong_query_vector_dimension():
    client = FakeMilvusClient()
    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    with pytest.raises(ValueError, match="1024"):
        vector_index.search(
            user_id="frank",
            vector=[0.1, 0.2],
            limit=2,
        )

    assert client.search_calls == []

class ExistingCollectionClient:
    def __init__(self):
        self.has_collection_calls = []

    def has_collection(self, collection_name: str) -> bool:
        self.has_collection_calls.append(collection_name)
        return True


def test_vector_index_keeps_existing_collection():
    client = ExistingCollectionClient()
    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    vector_index.ensure_collection()

    assert client.has_collection_calls == [
        "long_term_memory_vectors",
    ]

class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class MissingCollectionClient:
    def __init__(self):
        self.schema_kwargs = None
        self.schema = FakeSchema()
        self.index_params = FakeIndexParams()
        self.create_collection_calls = []

    def has_collection(self, collection_name: str) -> bool:
        return False

    def create_schema(self, **kwargs):
        self.schema_kwargs = kwargs
        return self.schema

    def prepare_index_params(self):
        return self.index_params

    def create_collection(
        self,
        collection_name: str,
        schema,
        index_params,
    ):
        self.create_collection_calls.append(
            (collection_name, schema, index_params)
        )


def test_vector_index_creates_schema_when_collection_is_missing():
    client = MissingCollectionClient()
    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    vector_index.ensure_collection()

    assert client.schema_kwargs == {
        "auto_id": False,
        "enable_dynamic_field": False,
    }
    assert client.schema.fields == [
        {
            "field_name": "memory_id",
            "datatype": DataType.INT64,
            "is_primary": True,
        },
        {
            "field_name": "user_id",
            "datatype": DataType.VARCHAR,
            "max_length": 64,
        },
        {
            "field_name": "embedding",
            "datatype": DataType.FLOAT_VECTOR,
            "dim": 1024,
        },
    ]
    assert client.index_params.indexes == [
        {
            "field_name": "embedding",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
        },
    ]
    assert client.create_collection_calls == [
        (
            "long_term_memory_vectors",
            client.schema,
            client.index_params,
        ),
    ]

def test_vector_index_ensures_collection_before_upsert():
    client = FakeMilvusClient()
    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    vector_index.upsert(
        memory_id=101,
        user_id="frank",
        vector=[0.1] * 1024,
    )

    assert client.has_collection_calls == [
        "long_term_memory_vectors",
    ]

def test_vector_index_ensures_collection_before_search():
    client = FakeMilvusClient()
    vector_index = MilvusMemoryVectorIndex(
        client=client,
        collection_name="long_term_memory_vectors",
    )

    vector_index.search(
        user_id="frank",
        vector=[0.1] * 1024,
        limit=2,
    )

    assert client.has_collection_calls == [
        "long_term_memory_vectors",
    ]

class RecordingMilvusClient:
    def __init__(self, uri: str):
        self.uri = uri


def test_build_milvus_client_passes_resolved_uri(monkeypatch):
    monkeypatch.setattr(
        milvus_memory,
        "MilvusClient",
        RecordingMilvusClient,
        raising=False,
    )

    client = build_milvus_client(
        uri="http://localhost:19530",
    )

    assert isinstance(client, RecordingMilvusClient)
    assert client.uri == "http://localhost:19530"
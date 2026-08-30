import os

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient
EMBEDDING_DIMENSION = 1024

def get_milvus_uri(uri: str | None = None) -> str:
    load_dotenv()

    resolved_uri = (
        uri
        if uri is not None
        else os.getenv("MILVUS_URI", "")
    )

    if not resolved_uri:
        raise ValueError("缺少 MILVUS_URI，无法连接 Milvus。")

    return resolved_uri

def build_milvus_client(
    uri: str | None = None,
) -> MilvusClient:
    return MilvusClient(
        uri=get_milvus_uri(uri),
    )

class MilvusMemoryVectorIndex:
    def __init__(
        self,
        client,
        collection_name: str,
    ):
        self.client = client
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        if self.client.has_collection(
            collection_name=self.collection_name,
        ):
            return
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="memory_id",
            datatype=DataType.INT64,
            is_primary=True,
        )
        schema.add_field(
            field_name="user_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIMENSION,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert(
        self,
        memory_id: int,
        user_id: str,
        vector: list[float],
    ) -> None:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding 向量维度必须是 {EMBEDDING_DIMENSION}。"
            )

        self.ensure_collection()

        self.client.upsert(
            collection_name=self.collection_name,
            data=[
                {
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "embedding": vector,
                }
            ],
        )

    def delete(self, memory_id: int) -> None:
        if memory_id < 1:
            raise ValueError("memory_id 必须至少为 1。")

        self.ensure_collection()

        self.client.delete(
            collection_name=self.collection_name,
            ids=[memory_id],
        )

    def search(
        self,
        user_id: str,
        vector: list[float],
        limit: int,
    ) -> list[int]:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding 向量维度必须是 {EMBEDDING_DIMENSION}。"
            )

        self.ensure_collection()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            filter=f'user_id == "{user_id}"',
            limit=limit,
        )

        return [
            result.id
            for result in results[0]
        ]

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI


MODEL_NAME = "text-embedding-v4"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

load_dotenv()


class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str | None = None, client: object | None = None):
        resolved_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
        if not resolved_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法生成 Embedding。")
        self.client = client or OpenAI(
            api_key=resolved_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=MODEL_NAME,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
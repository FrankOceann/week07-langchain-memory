import os
import json
import redis
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)


class BoundedChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        self.messages = self.messages[-(self.max_turns * 2) :]

    def clear(self) -> None:
        self.messages = []


class SessionHistoryStore:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self._histories: dict[str, BoundedChatMessageHistory] = {}

    def get(self, session_id: str) -> BoundedChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = BoundedChatMessageHistory(
                self.max_turns
            )

        return self._histories[session_id]

def build_redis_client(redis_url: str | None = None) -> redis.Redis:
    load_dotenv()

    resolved_url = (
        redis_url
        if redis_url is not None
        else os.getenv("REDIS_URL", "")
    )

    if not resolved_url:
        raise ValueError("缺少 REDIS_URL，无法创建 Redis 客户端。")

    return redis.Redis.from_url(resolved_url, decode_responses=True)

class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        client,
        session_id: str,
        max_turns: int,
        ttl_seconds: int,
    ):
        self.client = client
        self.key = f"week07:chat_history:{session_id}"
        self.max_messages = max_turns * 2
        self.ttl_seconds = ttl_seconds

    @property
    def messages(self) -> list[BaseMessage]:
        raw_messages = self.client.lrange(self.key, 0, -1)

        return [
            messages_from_dict([json.loads(raw_message)])[0]
            for raw_message in raw_messages
        ]

    def add_messages(self, messages: list[BaseMessage]) -> None:
        serialized_messages = [
            json.dumps(messages_to_dict([message])[0])
            for message in messages
        ]

        with self.client.pipeline() as pipeline:
            pipeline.rpush(self.key, *serialized_messages)
            pipeline.ltrim(self.key, -self.max_messages, -1)
            pipeline.expire(self.key, self.ttl_seconds)
            pipeline.execute()

    def clear(self) -> None:
        self.client.delete(self.key)

class RedisHistoryStore:
    def __init__(
        self,
        client,
        max_turns: int = 3,
        ttl_seconds: int = 1800,
    ):
        self.client = client
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds

    def get(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            client=self.client,
            session_id=session_id,
            max_turns=self.max_turns,
            ttl_seconds=self.ttl_seconds,
    )
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


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
# Conversational RAG Short-Term Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local interactive conversational RAG CLI with bounded, per-`session_id` memory. Every answer remains grounded in the current turn's Top-3 local documents.

**Architecture:** Preserve the existing Retriever. `app/memory.py` exclusively owns process-local bounded histories; `app/chat.py` owns prompt composition, a DashScope-compatible `ChatOpenAI` factory, and one RAG conversation function. `main.py` creates these dependencies once, then keeps them alive in an interactive loop.

**Tech Stack:** Python 3.10+, `langchain-core`, `langchain-openai`, OpenAI-compatible DashScope API, `python-dotenv`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-27-short-term-memory-design.md`

## Global Constraints

- First version uses process-local memory only; exit clears all sessions.
- Different `session_id` values must never share messages.
- Keep only the latest 3 complete question/answer turns (6 messages) per session.
- Retrieve the existing local Top-3 documents on every question and print their `source` metadata unchanged.
- The system prompt may use history to resolve references, but treats only the current-turn retrieved context as factual evidence.
- Use `qwen-plus` through DashScope's OpenAI-compatible URL, with private `DASHSCOPE_API_KEY` and optional `DASHSCOPE_BASE_URL`.
- All automated tests use fake embeddings and a fake chat runnable: no `.env`, HTTP, or API quota.
- Do not add Redis, a database, long-term memory, FastAPI, Docker, tools, or a persistent vector store.
- Windows CMD commands use `cd /d`. Never print, stage, commit, send, or screenshot a real `.env` key.

---

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `requirements.txt` | Modify | Add only `langchain-openai`. |
| `app/memory.py` | Create | Bounded chat-message history and per-session store. |
| `app/chat.py` | Create | Model creation, prompt/runnable, retrieval-to-answer flow. |
| `tests/test_memory.py` | Create | Isolation and exact turn-limit tests. |
| `tests/test_chat.py` | Create | Fake-only history/prompt/source/configuration tests. |
| `main.py` | Modify | Interactive in-process demo loop. |
| `README.md` | Modify after demonstration | Reproduction and limitations. |

### Task 1: Add the chat adapter dependency

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `from langchain_openai import ChatOpenAI` works after installation.

- [ ] **Step 1: Add the dependency**

Append exactly this line, without changing current requirements:

```text
langchain-openai
```

- [ ] **Step 2: Install and verify without calling a model**

Run in Windows CMD:

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "from langchain_openai import ChatOpenAI; print(ChatOpenAI.__name__)"
```

Expected: `ChatOpenAI`; this performs no API request.

### Task 2: Implement bounded, session-isolated history with TDD

**Files:**
- Create: `tests/test_memory.py`
- Create: `app/memory.py`

**Interfaces:**
- Consumes: `BaseChatMessageHistory`, `BaseMessage`.
- Produces: `BoundedChatMessageHistory(max_turns: int)` with `.messages`, `.add_messages(messages)`, `.clear()`; `SessionHistoryStore(max_turns: int = 3)` with `.get(session_id: str) -> BoundedChatMessageHistory`.

- [ ] **Step 1: Write the first failing test**

```python
from langchain_core.messages import AIMessage, HumanMessage

from app.memory import SessionHistoryStore


def test_session_history_store_keeps_sessions_isolated():
    store = SessionHistoryStore(max_turns=3)
    store.get("session-a").add_messages([HumanMessage("A 问题"), AIMessage("A 回答")])
    store.get("session-b").add_messages([HumanMessage("B 问题"), AIMessage("B 回答")])

    assert [item.content for item in store.get("session-a").messages] == ["A 问题", "A 回答"]
    assert [item.content for item in store.get("session-b").messages] == ["B 问题", "B 回答"]
```

- [ ] **Step 2: Confirm failure**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests/test_memory.py -q
```

Expected: failure because `app.memory` does not yet exist.

- [ ] **Step 3: Add the pruning test**

```python
def test_session_history_store_keeps_only_latest_complete_turns():
    store = SessionHistoryStore(max_turns=2)
    history = store.get("session-a")

    for index in range(3):
        history.add_messages([HumanMessage(f"Q{index}"), AIMessage(f"A{index}")])

    assert [item.content for item in history.messages] == ["Q1", "A1", "Q2", "A2"]
```

- [ ] **Step 4: Implement the smallest boundary**

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class BoundedChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        self.messages = self.messages[-(self.max_turns * 2):]

    def clear(self) -> None:
        self.messages = []


class SessionHistoryStore:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self._histories: dict[str, BoundedChatMessageHistory] = {}

    def get(self, session_id: str) -> BoundedChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = BoundedChatMessageHistory(self.max_turns)
        return self._histories[session_id]
```

Use an explicit store object rather than a module global; that keeps tests isolated and makes Redis a clean later swap.

- [ ] **Step 5: Verify and commit**

```cmd
.venv\Scripts\python.exe -m pytest tests/test_memory.py -q
git add app/memory.py tests/test_memory.py requirements.txt
git commit -m "feat: add bounded session message history"
```

Expected: 2 passed.

### Task 3: Build the offline-testable conversational RAG runnable with TDD

**Files:**
- Create: `tests/test_chat.py`
- Create: `app/chat.py`

**Interfaces:**
- Consumes: a retriever exposing `.invoke(question) -> list[Document]`, a chat runnable returning `AIMessage`, and `SessionHistoryStore.get`.
- Produces: `build_chat_model(api_key: str | None = None) -> ChatOpenAI`; `build_conversation_runnable(chat_model, history_store) -> RunnableWithMessageHistory`; `ask_question(question, session_id, retriever, conversation_runnable) -> tuple[str, list[str]]`.

- [ ] **Step 1: Write a failing history-and-context test**

Create a fake retriever that returns `Document(page_content="确认副作用前必须征得用户同意。", metadata={"source": "agent_safety.txt#chunk-0"})`. Create a fake chat runnable that records the received prompt messages and returns `AIMessage(content="假的回答")`. Call `ask_question()` twice with `session_id="a"`; assert that the second received prompt contains the first question, first answer, and current retrieved text.

- [ ] **Step 2: Write failing isolation, source, and missing-key tests**

For the same fakes: call session A, then session B, and assert B's prompt has no A message. Assert the returned sources are exactly `["agent_safety.txt#chunk-0"]`. Add:

```python
import pytest

from app.chat import build_chat_model


def test_build_chat_model_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        build_chat_model(api_key="")
```

- [ ] **Step 3: Confirm failure**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests/test_chat.py -q
```

Expected: import failure because `app.chat` does not yet exist.

- [ ] **Step 4: Implement the runnable and model factory**

Build the prompt as:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "只依据本轮检索资料回答；资料未覆盖时回答“资料不足”。历史仅用于理解指代，不能作为新事实来源。"),
    MessagesPlaceholder("history"),
    ("human", "本轮检索资料：\n{context}\n\n当前问题：{question}"),
])
chain = prompt | chat_model
runnable = RunnableWithMessageHistory(
    chain,
    history_store.get,
    input_messages_key="question",
    history_messages_key="history",
)
```

`ask_question()` must retrieve first, format each document as `[{source}]\n{page_content}`, invoke the runnable with `{"question": question, "context": context}` and `config={"configurable": {"session_id": session_id}}`, then return `(response.content, sources)`. It must not add documents themselves to chat history.

Call `load_dotenv()` in the model factory. Reject a missing key with `ValueError`; use the existing `DEFAULT_BASE_URL` fallback and return `ChatOpenAI(model="qwen-plus", api_key=..., base_url=..., temperature=0)`.

- [ ] **Step 5: Run offline tests and commit**

```cmd
.venv\Scripts\python.exe -m pytest tests/test_chat.py -q
.venv\Scripts\python.exe -m pytest -q
git add app/chat.py tests/test_chat.py
git commit -m "feat: add conversational rag runnable"
```

Expected: all original 8 tests plus new tests pass without an `.env`.

### Task 4: Replace the one-shot CLI with an interactive demo

**Files:**
- Modify: `main.py`
- Test: existing `tests/test_retriever.py` validation tests must remain green.

**Interfaces:**
- Consumes: current Retriever and the Task 2/3 interfaces.
- Produces: `main() -> int` that creates one retriever, history store, model, and runnable before a loop, so history survives for that process lifetime.

- [ ] **Step 1: Implement the loop**

Keep `validate_question()` unchanged. Build dependencies once. Read with `input("你：")`; exit on `exit`, `quit`, or `退出`, and return 0 on `EOFError`. On blank input, print `错误：问题不能为空。` and continue. Call:

```python
answer, sources = ask_question(
    question,
    session_id="demo-session",
    retriever=retriever,
    conversation_runnable=conversation_runnable,
)
```

Print `助手：{answer}` and each `=== {source} ===`. At setup, retain the current safe `ValueError -> 错误：...` stderr / exit 2 handling. Never print tracebacks, headers, or environment values.

- [ ] **Step 2: Verify and commit**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest -q
git add main.py
git commit -m "feat: add interactive conversational rag cli"
```

Expected: full offline suite passes.

### Task 5: Real multi-turn demonstration using only private configuration

**Files:**
- Modify locally only, if needed: `.env`
- Do not stage: `.env`

- [ ] **Step 1: Verify secret protection**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git check-ignore -v .env
```

Expected: the ignore rule is shown.

- [ ] **Step 2: Configure privately, only if needed**

```cmd
copy .env.example .env
notepad .env
```

Use a key and matching region/workspace base URL. Never copy the file contents into chat, output, Git, or a screenshot.

- [ ] **Step 3: Demonstrate a follow-up in one process**

```cmd
.venv\Scripts\python.exe main.py
```

Ask a grounded first question about `data/agent_safety.txt`, then `那为什么？`. Confirm both answers print source labels and that the second resolves the reference from the first turn while using newly retrieved current-turn context. Type `退出`, then run `git status --short`; `.env` must not appear.

### Task 6: Document, final-verify, commit, and push

**Files:**
- Modify: `README.md`
- Modify: the short-term-memory spec only if actual behavior differs from approved design.

- [ ] **Step 1: Update README after the successful demo**

Add “第二节：对话式 RAG 短期记忆”: data flow, `RunnableWithMessageHistory`, `session_id` isolation, 3-turn bound, interactive CMD command, `退出`, private config, fake-only tests, and all explicit non-goals.

- [ ] **Step 2: Record the limitation and next lesson**

State that history lives only in one process and disappears on restart. State that the next lesson migrates the existing `session_id` boundary to Redis with TTL; add no Redis code.

- [ ] **Step 3: Final evidence, commit, and push**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest -q
git status --short
git add README.md docs/superpowers/specs/2026-08-27-short-term-memory-design.md docs/superpowers/plans/2026-08-27-short-term-memory.md
git commit -m "docs: document conversational rag memory"
git push origin main
```

If the spec did not change, omit it from `git add`. Inspect staged files before committing and never stage `.env`.

## Plan Self-Review

- Tasks 2–4 cover all approved behavior: current-turn Top-3 retrieval, source preservation, session isolation, bounded memory, grounding prompt, interactive lifecycle, and safe configuration errors.
- Task 3 gives deterministic offline tests; Task 5 is the only real API exercise; Task 6 records the proven result and Redis handoff.
- No task introduces an excluded subsystem.
- Type flow is consistent: `SessionHistoryStore.get` is the history factory, and `ask_question()` returns `(str, list[str])` for `main.py`.


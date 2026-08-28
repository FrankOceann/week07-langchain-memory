# Redis 会话短期记忆 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将对话式 RAG 的会话历史从进程内字典迁移到带滑动 TTL 的 Redis List，且不改变检索与对话 Runnable 的行为。

**Architecture:** `RedisChatMessageHistory` 实现 LangChain 的 `BaseChatMessageHistory`，将一条序列化后的 LangChain 消息存为 Redis List 的一个元素。`RedisHistoryStore` 按 `session_id` 创建该历史对象；写入使用 pipeline 依次追加、裁剪并刷新 TTL，`RunnableWithMessageHistory` 仍通过 `.get(session_id)` 取得历史。

**Tech Stack:** Python 3.10、LangChain Core、`redis-py`、`fakeredis`、Docker Compose、Redis、pytest。

**Spec:** `docs/superpowers/specs/2026-08-27-redis-short-term-memory-design.md`

## Global Constraints

- Windows CMD 命令一律先使用 `cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"`。
- 代码、测试、依赖、配置和 Docker Compose 均由学习者自己修改；Codex 只能先说明改哪里、为什么、如何验证。
- Git 暂存、提交与推送由学习者自己执行；绝不提交 `.env` 或真实 API Key。
- Redis 仅保存短期会话历史：默认最多 3 轮（6 条消息）、默认 TTL 1800 秒、每次写入刷新 TTL。
- 每轮仍由现有 Retriever 检索资料；Redis 不替代检索器、向量库、长期记忆或数据库。
- 本节继续使用 `RunnableWithMessageHistory`；不迁移 LangGraph，不处理其现有弃用警告。

---

## 文件结构

- `requirements.txt`：记录运行 Redis 客户端和离线 Fake Redis 所需的依赖。
- `compose.yaml`：只启动本机 Redis 服务，不容器化 Python 应用。
- `.env.example`：提供不带凭据的 `REDIS_URL` 示例。
- `app/memory.py`：保留旧进程内实现作为对比，新增 Redis 客户端工厂、Redis 消息历史和 Redis 历史工厂。
- `app/chat.py`：不改变 `build_conversation_runnable()` 接口；它继续接受任何提供 `.get(session_id)` 的历史工厂。
- `main.py`：由 `RedisHistoryStore` 替代 `SessionHistoryStore`，使实际 CLI 使用 Redis。
- `tests/test_memory.py`：使用 `fakeredis` 测试隔离、裁剪、TTL、清空和配置错误。
- `tests/test_chat.py`：把现有同会话多轮测试的历史工厂替换为 Redis Fake，证明 Runnable 不需要改动。
- `README.md`：记录 Docker Compose、私有 `REDIS_URL`、测试、真实演示与已知边界。
- `D:\桌面\所有codex项目\AI agent 开发\python 学习\README.md`：更新“新窗口接续提示”为 Redis 已完成后的真实状态。

### Task 1: 准备 Redis 开发依赖与本地服务定义

**Files:**
- Modify: `requirements.txt`
- Create: `compose.yaml`
- Modify: `.env.example`

**Interfaces:**
- Consumes: 私有 `.env` 已有 DashScope 配置；Docker Desktop 可用。
- Produces: `redis`、`fakeredis` 可导入；`docker compose up -d redis` 可启动本机 `localhost:6379` Redis；应用可从 `REDIS_URL` 获得连接地址。

- [ ] **Step 1: 在 `requirements.txt` 末尾新增两个依赖**

```text
redis
fakeredis
```

`redis` 是 Python 连接 Redis 的正式客户端；`fakeredis` 只在 pytest 中模拟 Redis，保证离线测试不需要 Docker。

- [ ] **Step 2: 安装并确认两个依赖能导入**

在 Windows CMD 运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import fakeredis, redis; print(redis.__version__)"
```

Expected: 最后一行输出一个 `redis` 版本号且没有 `ModuleNotFoundError`。

- [ ] **Step 3: 创建 `compose.yaml`，只定义 Redis 服务**

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
```

`6379:6379` 让本机 Python 通过 `localhost:6379` 访问容器；`appendonly yes` 使 Redis 容器重启时可恢复自身数据。它不改变“TTL 到期会删除会话”的规则。

- [ ] **Step 4: 验证 Compose 文件语法，不启动服务**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
docker compose config
```

Expected: 输出解析后的 `services.redis` 配置，且命令退出码为 0。

- [ ] **Step 5: 在 `.env.example` 追加无凭据 Redis 示例**

```dotenv
REDIS_URL=redis://localhost:6379/0
```

不要在 `.env.example` 写真实密码或企业地址。稍后把同一行仅复制到自己私有、被忽略的 `.env`。

- [ ] **Step 6: 本任务检查**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git diff --check
```

Expected: 没有输出；此时尚未运行 pytest，因为 Redis 历史类和相应测试还不存在。

### Task 2: 先为 Redis 消息历史写失败测试

**Files:**
- Modify: `tests/test_memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `fakeredis.FakeRedis(decode_responses=True)`、`HumanMessage`、`AIMessage`。
- Produces: 对 `RedisHistoryStore(client, max_turns=3, ttl_seconds=30)` 与 `RedisChatMessageHistory` 的行为约束；这些名称由下一任务实现。

- [ ] **Step 1: 在 `tests/test_memory.py` 顶部新增导入**

```python
import fakeredis
import pytest

from app.memory import (
    RedisHistoryStore,
    build_redis_client,
)
```

保留已有 `SessionHistoryStore` 导入和现有两条进程内历史测试，作为旧实现的对比基线。

- [ ] **Step 2: 追加会话隔离与轮数裁剪的失败测试**

```python
def test_redis_history_keeps_sessions_isolated_and_trims_turns():
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisHistoryStore(
        client,
        max_turns=2,
        ttl_seconds=30,
    )

    for index in range(3):
        store.get("session-a").add_messages(
            [HumanMessage(f"A-Q{index}"), AIMessage(f"A-A{index}")]
        )
    store.get("session-b").add_messages(
        [HumanMessage("B-Q0"), AIMessage("B-A0")]
    )

    assert [item.content for item in store.get("session-a").messages] == [
        "A-Q1",
        "A-A1",
        "A-Q2",
        "A-A2",
    ]
    assert [item.content for item in store.get("session-b").messages] == [
        "B-Q0",
        "B-A0",
    ]
```

- [ ] **Step 3: 追加 TTL、清空和缺少 URL 的失败测试**

```python
def test_redis_history_refreshes_ttl_and_clear_only_removes_its_session():
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisHistoryStore(client, max_turns=3, ttl_seconds=30)
    history_a = store.get("session-a")
    history_b = store.get("session-b")

    history_a.add_messages([HumanMessage("A-Q"), AIMessage("A-A")])
    history_b.add_messages([HumanMessage("B-Q"), AIMessage("B-A")])

    assert client.ttl(history_a.key) > 0
    history_a.clear()

    assert history_a.messages == []
    assert [item.content for item in history_b.messages] == ["B-Q", "B-A"]


def test_build_redis_client_rejects_missing_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(ValueError, match="REDIS_URL"):
        build_redis_client(redis_url="")
```

`history_a.key` 是有意暴露给测试和真实 TTL 演示检查的只读实例属性。

- [ ] **Step 4: 运行新测试，确认失败原因是尚未实现 Redis 接口**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_memory.py -q
```

Expected: 测试收集失败，错误包含 `cannot import name 'RedisHistoryStore'` 或 `cannot import name 'build_redis_client'`。这是 TDD 的预期红灯，不应为了让它通过而删除测试。

### Task 3: 实现 Redis 客户端和 LangChain 消息历史适配器

**Files:**
- Modify: `app/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `redis.Redis` 或 `fakeredis.FakeRedis` 客户端，均需提供 `lrange`、`pipeline`、`delete`、`ttl`。
- Produces:
  - `build_redis_client(redis_url: str | None = None) -> redis.Redis`
  - `RedisChatMessageHistory(client, session_id: str, max_turns: int, ttl_seconds: int)`
  - `RedisHistoryStore(client, max_turns: int = 3, ttl_seconds: int = 1800)`，提供 `.get(session_id)`。

- [ ] **Step 1: 在 `app/memory.py` 顶部增加 Redis、环境变量和消息序列化导入**

```python
import json
import os

import redis
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)
```

保留原有 `BaseChatMessageHistory` 导入。`messages_to_dict` 与 `messages_from_dict` 确保 Redis 中保存了消息类型和内容，而不是只保存无法区分角色的字符串。

- [ ] **Step 2: 在 `SessionHistoryStore` 类之后实现客户端工厂**

```python
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
```

不要在此函数中执行 `ping()`：创建客户端不应隐式联网；真实 CLI 的第一次读写会自然暴露连接错误。

- [ ] **Step 3: 在同一文件实现 `RedisChatMessageHistory`**

```python
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
```

`lrange(..., 0, -1)` 返回整个 List；`LTRIM key -6 -1` 保留最后 6 条。即使一次传入多条消息，仍只执行一组 pipeline 操作并刷新一次 TTL。

- [ ] **Step 4: 在同一文件实现 `RedisHistoryStore`**

```python
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
```

不在 Python 中再保存 `_histories` 字典：Redis 本身就是跨进程的共享状态源。

- [ ] **Step 5: 运行 Redis 历史测试，确认变绿**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_memory.py -q
```

Expected: 所有 `tests/test_memory.py` 测试通过；旧进程内历史测试和新 Redis Fake 测试均通过。

### Task 4: 让对话链与实际 CLI 使用 Redis 历史

**Files:**
- Modify: `tests/test_chat.py`
- Modify: `main.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `RedisHistoryStore(client, max_turns=3, ttl_seconds=1800)`，其 `.get(session_id)` 与原 `SessionHistoryStore` 相同。
- Produces: 现有 `build_conversation_runnable()` 不变；真实 CLI 使用从 `.env` 创建的 Redis 客户端。

- [ ] **Step 1: 在 `tests/test_chat.py` 写出迁移后的失败测试改动**

将：

```python
from app.memory import SessionHistoryStore
```

替换为：

```python
import fakeredis

from app.memory import RedisHistoryStore
```

并将测试中创建 Runnable 的第二个参数替换为：

```python
RedisHistoryStore(
    fakeredis.FakeRedis(decode_responses=True),
    max_turns=3,
    ttl_seconds=30,
)
```

这条既有测试仍应断言第二轮 prompt 包含第一轮问题、第一轮假回答和本轮 Retriever 返回的资料。它证明 Redis 后端没有破坏 `RunnableWithMessageHistory` 的多轮行为。

- [ ] **Step 2: 运行该对话测试，观察当前代码是否已满足接口**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_chat.py -q
```

Expected: Task 3 完成后这一步应通过；如果失败，先保留完整 Traceback，再检查 Redis 消息反序列化是否恢复了 `HumanMessage` 与 `AIMessage` 类型。

- [ ] **Step 3: 修改 `main.py` 的导入和历史工厂**

将：

```python
from app.memory import SessionHistoryStore
```

替换为：

```python
from app.memory import RedisHistoryStore, build_redis_client
```

将：

```python
history_store = SessionHistoryStore(max_turns=3)
```

替换为：

```python
history_store = RedisHistoryStore(
    build_redis_client(),
    max_turns=3,
    ttl_seconds=1800,
)
```

保留已有 `try/except ValueError`。因此缺少 `REDIS_URL` 时，CLI 会输出清晰错误并返回状态码 2；不允许悄悄回退到进程内记忆。

- [ ] **Step 4: 运行离线完整测试**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest -q
```

Expected: 所有测试通过，且不需要 Docker、Redis 网络服务或真实 DashScope API Key。现有 `RunnableWithMessageHistory` 弃用警告可保留并在 README 记录。

### Task 5: 启动真实 Redis 并做 TTL、重启与错误路径演示

**Files:**
- Modify: 私有 `.env`（不得提交）
- Test: 手动 Redis/CLI 验收，不新增业务代码

**Interfaces:**
- Consumes: `compose.yaml` 的 Redis 服务、`.env` 的 `REDIS_URL`、`main.py` 的 Redis 历史工厂。
- Produces: 真实 Redis 会话历史、正数 TTL、Python 重启后继续同一会话的证据。

- [ ] **Step 1: 只在私有 `.env` 追加 Redis 地址**

```dotenv
REDIS_URL=redis://localhost:6379/0
```

不要展示或提交 `.env` 中的 API Key。先验证其仍被忽略：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git check-ignore -v .env
```

Expected: 输出 `.gitignore` 中匹配 `.env` 的规则。

- [ ] **Step 2: 启动 Redis 并检查运行状态**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
docker compose up -d redis
docker compose ps
docker compose exec redis redis-cli PING
```

Expected: Redis 服务状态为 running，最后一行是 `PONG`。

- [ ] **Step 3: 运行两轮真实对话**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe main.py redis-demo
```

依次输入第一问（例如“如何确认副作用操作？”）与第二问（例如“那为什么？”），再输入 `退出`。第二轮应能理解“那”指上一轮主题；每轮仍输出本轮 Retriever 来源。

- [ ] **Step 4: 验证键和 TTL**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
docker compose exec redis redis-cli TTL week07:chat_history:redis-demo
docker compose exec redis redis-cli LRANGE week07:chat_history:redis-demo 0 -1
```

Expected: `TTL` 返回正数且小于等于 1800；`LRANGE` 返回序列化消息文本。不要把输出中可能含有的个人对话内容提交到 Git。

- [ ] **Step 5: 验证 Python 重启后历史仍在**

再次运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe main.py redis-demo
```

输入“那为什么？”，确认模型能从 Redis 读取之前的第一轮主题；然后输入 `退出`。

- [ ] **Step 6: 验证 Redis 不可用时不会静默回退**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
docker compose stop redis
.venv\Scripts\python.exe main.py redis-demo
docker compose start redis
```

Expected: Redis 停止期间的 CLI 首次读写显示连接失败，不会改用 `SessionHistoryStore`；最后一条命令恢复 Redis 服务。

### Task 6: 更新文档并由学习者完成 Git 交付

**Files:**
- Modify: `README.md`
- Modify: `D:\桌面\所有codex项目\AI agent 开发\python 学习\README.md`
- Create/Modify: `docs/superpowers/plans/2026-08-27-redis-short-term-memory.md`（勾选完成项）

**Interfaces:**
- Consumes: Task 4 的完整离线测试结果和 Task 5 的真实验收结果。
- Produces: 可复现的 Redis 学习记录与下一节“长期记忆”接续提示。

- [ ] **Step 1: 更新项目 `README.md`**

新增 Redis 一节，必须说明：Docker Compose 仅运行 Redis；`REDIS_URL` 的 `.env` 配置；键命名、3 轮上限、1800 秒滑动 TTL；Docker 启动/停止命令；离线测试命令；真实演示的“同一 session_id、Python 重启仍有历史”；Redis 不是长期记忆；`RunnableWithMessageHistory` 的已知弃用警告。

- [ ] **Step 2: 更新总学习 README 的“新窗口接续提示”**

将 Redis 标记为完成，写入实际测试数、实际提交哈希和真实演示结论；将下一步明确为“长期记忆：复用 Week05 SQLModel Memory API”。不要把 Docker Compose 的本机地址误写成生产部署。

- [ ] **Step 3: 提交前由学习者自己核对敏感文件和完整测试**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git status --short
git check-ignore -v .env
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: `.env` 被忽略；完整 pytest 通过；`git diff --check` 无输出；待提交文件不包含 `.env`、`.venv`、缓存或真实对话输出。

- [ ] **Step 4: 由学习者自己暂存、提交和推送**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git add requirements.txt compose.yaml .env.example app\memory.py main.py tests\test_memory.py tests\test_chat.py README.md docs\superpowers\specs\2026-08-27-redis-short-term-memory-design.md docs\superpowers\plans\2026-08-27-redis-short-term-memory.md
git commit -m "feat: move chat history to redis"
git push origin main
git status -sb
```

Expected: 最后显示 `main...origin/main`，没有未提交改动。执行者必须在 `git add` 前逐项确认文件存在且不含敏感内容；如果最终文件列表不同，应调整 `git add` 的精确路径，而不是使用 `git add .`。

## Plan Self-Review

- Spec coverage：Task 1 覆盖 Docker Compose 与环境变量；Task 2–3 覆盖 Redis List、消息序列化、隔离、裁剪、TTL、清空和错误；Task 4 保持 Runnable/Retriever 行为并切换 CLI；Task 5 覆盖真实 Docker、TTL、重启和连接失败；Task 6 覆盖文档和学习者 Git 交付。
- Placeholder scan：计划不含 `TBD`、`TODO` 或“稍后实现”等未定义步骤。
- Type consistency：`RedisHistoryStore.get(session_id)` 与现有 `RunnableWithMessageHistory` 所需历史工厂一致；测试、`main.py` 和实现均使用 `max_turns`、`ttl_seconds`、`build_redis_client` 这组统一名称。

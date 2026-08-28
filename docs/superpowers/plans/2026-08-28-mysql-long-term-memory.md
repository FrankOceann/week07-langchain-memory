# Week07 MySQL 结构化长期记忆 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 Docker MySQL、SQLAlchemy 和 Alembic 实现按用户隔离、可逻辑停用的结构化长期记忆，并让现有聊天 CLI 只读它。

**Architecture:** MySQL 的 `long_term_memories` 是唯一权威来源；Repository 隔离读写。每次聊天先读有效长期记忆，再与 Redis 历史、本轮 RAG 资料一起交给已有 `qwen-plus`；Redis、Retriever 和 Runnable 保持原职责。

**Tech Stack:** Python 3.10、MySQL 8.4、Docker Compose、SQLAlchemy 2.x、Alembic、PyMySQL、Redis、LangChain Core、pytest。

**Spec:** `docs/superpowers/specs/2026-08-28-mysql-long-term-memory-design.md`

## Global Constraints

- Windows CMD 命令一律先使用 `cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"`。
- 代码、测试、依赖、Docker Compose 与配置均由学习者修改；Git 暂存、提交和推送也由学习者执行。
- 不提交 `.env`、真实密码/API Key、`.venv`、Docker 数据卷或 `pytest-of-wurunnan/`。
- Redis 只保存按 `session_id` 隔离的 3 轮短期历史；MySQL 是按 `user_id` 隔离的长期记忆权威源。
- 模型不能自动写入/修改长期记忆；MySQL 失败时明确中止本轮，不能静默跳过。
- 本计划不引入 Milvus、FastAPI、认证、队列、Outbox、LangGraph 或 MySQL 高可用。

---

## 文件结构

- `requirements.txt`、`compose.yaml`、`.env.example`：MySQL 开发依赖和本机容器配置。
- `app/models.py`：`Base`、`LongTermMemory` 数据模型。
- `app/database.py`：URL、Engine、Session 工厂；不建表。
- `app/long_term_memory.py`：Repository、分类校验和 Prompt 渲染。
- `alembic.ini`、`migrations/`：唯一的 schema 演进方式。
- `app/chat.py`：读取长期记忆并注入 Prompt。
- `main.py`：`chat`、`memory add/list/deactivate` 命令解析和依赖组装。
- `tests/test_database.py`、`tests/test_long_term_memory.py`、`tests/test_main.py`：离线测试。
- `tests/test_chat.py`：长期记忆 Prompt 与故障路径测试。

### Task 1: 配置 MySQL 依赖与 Docker Compose

**Files:**
- Modify: `requirements.txt`
- Modify: `compose.yaml`
- Modify: `.env.example`

**Produces:** SQLAlchemy、Alembic、PyMySQL 可导入；本机可通过 Docker 启动 MySQL 8.4，应用仅用非 root 账号连接。

- [ ] **Step 1: 在 `requirements.txt` 末尾新增依赖**

```text
SQLAlchemy
alembic
PyMySQL
```

- [ ] **Step 2: 安装并验证导入**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import alembic, pymysql, sqlalchemy; print(sqlalchemy.__version__)"
```

Expected: 输出 SQLAlchemy 版本，无 `ModuleNotFoundError`。

- [ ] **Step 3: 在 `.env.example` 追加无敏感变量**

```dotenv
MYSQL_DATABASE=agent_memory
MYSQL_USER=agent_app
MYSQL_PASSWORD=
MYSQL_ROOT_PASSWORD=
MYSQL_URL=
```

私有 `.env` 中的 URL 格式是 `mysql+pymysql://agent_app:<URL编码密码>@localhost:3306/agent_memory`；密码含 `@`、`:`、`/` 时先 URL 编码。

- [ ] **Step 4: 在现有 Redis 服务后加入 MySQL 与命名卷**

```yaml
  mysql:
    image: mysql:8.4
    environment:
      MYSQL_DATABASE: <来自私有 .env>
      MYSQL_USER: <来自私有 .env>
      MYSQL_PASSWORD: <来自私有 .env>
      MYSQL_ROOT_PASSWORD: <来自私有 .env>
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD-SHELL", "mysqladmin ping -h localhost --silent"]
      interval: 5s
      timeout: 5s
      retries: 20

volumes:
  mysql_data:
```

填写时使用 Compose 的变量插值，将每个 `<来自私有 .env>` 替换为对应环境变量；应用运行时不用 root。命名卷防止容器重建丢失数据。

- [ ] **Step 5: 验证但暂不启动**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
docker compose config
git diff --check
```

Expected: 有 `redis`、`mysql` 两个服务，且无 diff 格式问题。

### Task 2: TDD 实现数据库 URL、Engine 与表模型

**Files:**
- Create: `tests/test_database.py`
- Create: `app/models.py`
- Create: `app/database.py`

**Produces:** `Base`、`LongTermMemory`、`get_mysql_url()`、`build_engine()`、`build_session_factory()`。

- [ ] **Step 1: 先写失败测试 `tests/test_database.py`**

```python
import pytest
from sqlalchemy import inspect

from app.database import build_engine, get_mysql_url
from app.models import LongTermMemory


def test_get_mysql_url_rejects_missing_url(monkeypatch):
    monkeypatch.delenv("MYSQL_URL", raising=False)
    with pytest.raises(ValueError, match="MYSQL_URL"):
        get_mysql_url(mysql_url="")


def test_long_term_memory_model_declares_expected_columns():
    engine = build_engine("sqlite+pysqlite:///:memory:")
    LongTermMemory.metadata.create_all(engine)
    columns = {
        column["name"]
        for column in inspect(engine).get_columns("long_term_memories")
    }
    assert columns == {
        "id", "user_id", "category", "content", "source",
        "is_active", "created_at", "updated_at",
    }
```

- [ ] **Step 2: 运行红灯**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_database.py -q
```

Expected: `No module named 'app.database'` 或 `app.models`。

- [ ] **Step 3: 创建 `app/models.py`**

```python
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class LongTermMemory(Base):
    __tablename__ = "long_term_memories"
    __table_args__ = (
        Index(
            "ix_long_term_memories_user_active_category_updated",
            "user_id", "is_active", "category", "updated_at",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now, onupdate=utc_now
    )
```

- [ ] **Step 4: 创建 `app/database.py`**

```python
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_mysql_url(mysql_url: str | None = None) -> str:
    load_dotenv()
    resolved_url = mysql_url if mysql_url is not None else os.getenv("MYSQL_URL", "")
    if not resolved_url:
        raise ValueError("缺少 MYSQL_URL，无法连接长期记忆数据库。")
    return resolved_url


def build_engine(mysql_url: str | None = None):
    return create_engine(get_mysql_url(mysql_url), pool_pre_ping=True)


def build_session_factory(mysql_url: str | None = None):
    return sessionmaker(bind=build_engine(mysql_url), expire_on_commit=False)
```

不得调用 `Base.metadata.create_all()`；schema 由迁移建立。

- [ ] **Step 5: 验证变绿**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_database.py -q
```

Expected: `2 passed`，没有 Docker 连接。

### Task 3: TDD 实现 Repository、用户隔离与逻辑停用

**Files:**
- Create: `tests/test_long_term_memory.py`
- Create: `app/long_term_memory.py`

**Produces:** `SQLAlchemyLongTermMemoryRepository`，含 `.add()`、`.list_active()`、`.deactivate()` 和 `render_long_term_memories()`。

- [ ] **Step 1: 写离线 Repository 失败测试**

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.long_term_memory import SQLAlchemyLongTermMemoryRepository
from app.models import Base


@pytest.fixture
def repository():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SQLAlchemyLongTermMemoryRepository(
        sessionmaker(bind=engine, expire_on_commit=False)
    )


def test_repository_isolates_users_and_hides_deactivated_memory(repository):
    old = repository.add("frank", "preference", "旧偏好")
    repository.add("frank", "profile", "正在学习 AI Agent")
    repository.add("alice", "preference", "Alice 的偏好")
    assert repository.deactivate(old.id) is True

    memories = repository.list_active("frank")

    assert [(item.user_id, item.content) for item in memories] == [
        ("frank", "正在学习 AI Agent")
    ]


def test_repository_filters_category_limits_and_rejects_invalid_input(repository):
    repository.add("frank", "preference", "偏好一")
    repository.add("frank", "profile", "资料一")
    repository.add("frank", "preference", "偏好二")

    assert [item.content for item in repository.list_active(
        "frank", category="preference", limit=1
    )] == ["偏好二"]
    with pytest.raises(ValueError, match="category"):
        repository.add("frank", "unknown", "内容")
    assert repository.deactivate(999) is False
```

- [ ] **Step 2: 运行红灯**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_long_term_memory.py -q
```

Expected: `No module named 'app.long_term_memory'`。

- [ ] **Step 3: 创建最小 Repository**

```python
from collections.abc import Callable

from sqlalchemy import select

from app.models import LongTermMemory

ALLOWED_CATEGORIES = {"preference", "profile", "fact"}


class SQLAlchemyLongTermMemoryRepository:
    def __init__(self, session_factory: Callable):
        self.session_factory = session_factory

    def add(self, user_id, category, content, source="user_confirmed"):
        if category not in ALLOWED_CATEGORIES:
            raise ValueError("category 必须是 preference、profile 或 fact。")
        if not user_id.strip() or not content.strip():
            raise ValueError("user_id 和 content 不能为空。")
        with self.session_factory() as session:
            memory = LongTermMemory(
                user_id=user_id.strip(), category=category,
                content=content.strip(), source=source,
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            return memory

    def list_active(self, user_id, category=None, limit=5):
        if category is not None and category not in ALLOWED_CATEGORIES:
            raise ValueError("category 必须是 preference、profile 或 fact。")
        if limit < 1:
            raise ValueError("limit 必须至少为 1。")
        statement = select(LongTermMemory).where(
            LongTermMemory.user_id == user_id,
            LongTermMemory.is_active.is_(True),
        )
        if category is not None:
            statement = statement.where(LongTermMemory.category == category)
        statement = statement.order_by(
            LongTermMemory.updated_at.desc(), LongTermMemory.id.desc()
        ).limit(limit)
        with self.session_factory() as session:
            return list(session.scalars(statement))

    def deactivate(self, memory_id):
        with self.session_factory() as session:
            memory = session.get(LongTermMemory, memory_id)
            if memory is None or not memory.is_active:
                return False
            memory.is_active = False
            session.commit()
            return True


def render_long_term_memories(memories):
    if not memories:
        return "无已确认长期记忆。"
    return "\n".join(
        f"[memory:{memory.id}] ({memory.category}) {memory.content}"
        for memory in memories
    )
```

- [ ] **Step 4: 追加 Prompt 渲染测试并验证**

```python
def test_render_long_term_memories(repository):
    assert render_long_term_memories([]) == "无已确认长期记忆。"
    memory = repository.add("frank", "preference", "使用中文")
    assert render_long_term_memories([memory]) == (
        f"[memory:{memory.id}] (preference) 使用中文"
    )
```

补充导入 `render_long_term_memories`，运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_long_term_memory.py -q
```

Expected: 全部通过，不连接 MySQL。

### Task 4: TDD 将长期记忆注入聊天 Prompt 并保持严格故障策略

**Files:**
- Modify: `tests/test_chat.py`
- Modify: `app/chat.py`

**Produces:** `ask_question(..., user_id, long_term_memory_repository)`；MySQL 查询失败时模型零调用。

- [ ] **Step 1: 在 `tests/test_chat.py` 新增 Fake Repository**

```python
from types import SimpleNamespace


class FakeLongTermMemoryRepository:
    def __init__(self, memories=None, error=None):
        self.memories = memories or []
        self.error = error

    def list_active(self, user_id, category=None, limit=5):
        if self.error is not None:
            raise self.error
        return [item for item in self.memories if item.user_id == user_id]
```

- [ ] **Step 2: 给已有两次 `ask_question()` 调用增加失败参数**

```python
user_id="frank",
long_term_memory_repository=FakeLongTermMemoryRepository([
    SimpleNamespace(
        id=101, user_id="frank", category="preference",
        content="使用中文分步骤说明。",
    ),
    SimpleNamespace(
        id=102, user_id="alice", category="preference",
        content="不得泄露给 frank。",
    ),
]),
```

增加断言：

```python
assert "[memory:101] (preference) 使用中文分步骤说明。" in second_prompt_text
assert "不得泄露给 frank。" not in second_prompt_text
```

- [ ] **Step 3: 添加失败路径测试**

```python
def test_long_term_memory_failure_does_not_call_chat_model():
    model_calls = []

    def fake_response(prompt_value):
        model_calls.append(prompt_value)
        return AIMessage(content="不应生成")

    runnable = build_conversation_runnable(
        RunnableLambda(fake_response),
        RedisHistoryStore(fakeredis.FakeRedis(decode_responses=True), 3, 30),
    )

    with pytest.raises(ConnectionError, match="MySQL 不可用"):
        ask_question(
            "如何确认副作用操作？",
            session_id="session-a",
            user_id="frank",
            retriever=FakeRetriever(),
            conversation_runnable=runnable,
            long_term_memory_repository=FakeLongTermMemoryRepository(
                error=ConnectionError("MySQL 不可用")
            ),
        )
    assert model_calls == []
```

- [ ] **Step 4: 运行红灯**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_chat.py -q
```

Expected: 报 `unexpected keyword argument 'user_id'` 或缺少 `long_term_memory_repository`。

- [ ] **Step 5: 修改 `app/chat.py`**

先导入：

```python
from app.long_term_memory import render_long_term_memories
```

将系统 Prompt 替换为包含 `{long_term_memory}` 的文本：

```python
"本轮检索资料用于知识事实；资料未覆盖时回答“资料不足”。"
"历史仅用于理解指代，不能作为新事实来源。"
"长期记忆仅用于已确认的用户偏好、资料和事实，不能覆盖本轮检索资料，"
"也不能自行新增、修改或停用记忆。\n\n"
"已确认长期记忆：\n{long_term_memory}"
```

把函数签名改为：

```python
def ask_question(
    question, session_id, user_id, retriever,
    conversation_runnable, long_term_memory_repository,
) -> tuple[str, list[str]]:
```

在 Retriever 前读取：

```python
memories = long_term_memory_repository.list_active(user_id)
long_term_memory = render_long_term_memories(memories)
```

把 Runnable 输入改为：

```python
{
    "question": question,
    "context": context,
    "long_term_memory": long_term_memory,
}
```

Repository 先于模型运行，异常自然中止，符合严格模式。

- [ ] **Step 6: 验证变绿**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_chat.py -q
```

Expected: 现有 Redis 多轮、RAG 来源与新增长期记忆断言均通过。

### Task 5: TDD 改造 CLI 为显式聊天和人工记忆管理命令

**Files:**
- Create: `tests/test_main.py`
- Modify: `main.py`

**Produces:** `build_parser()`、`main(argv: list[str] | None = None) -> int`、`chat` 与 `memory add/list/deactivate`。

- [ ] **Step 1: 写 CLI 解析失败测试**

```python
import pytest

from main import build_parser


def test_chat_command_requires_session_id_and_user_id():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--session-id", "redis-demo"])

    args = parser.parse_args(
        ["chat", "--session-id", "redis-demo", "--user-id", "frank"]
    )
    assert (args.command, args.session_id, args.user_id) == (
        "chat", "redis-demo", "frank"
    )


def test_memory_add_requires_valid_category():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "memory", "add", "--user-id", "frank",
            "--category", "unknown", "--content", "内容",
        ])
```

- [ ] **Step 2: 运行红灯**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_main.py -q
```

Expected: `cannot import name 'build_parser' from 'main'`。

- [ ] **Step 3: 在 `main.py` 添加解析器**

```python
import argparse
from app.database import build_session_factory
from app.long_term_memory import (
    ALLOWED_CATEGORIES,
    SQLAlchemyLongTermMemoryRepository,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat")
    chat.add_argument("--session-id", required=True)
    chat.add_argument("--user-id", required=True)

    memory = commands.add_parser("memory")
    memory_commands = memory.add_subparsers(
        dest="memory_command", required=True
    )
    add = memory_commands.add_parser("add")
    add.add_argument("--user-id", required=True)
    add.add_argument("--category", choices=sorted(ALLOWED_CATEGORIES), required=True)
    add.add_argument("--content", required=True)
    list_command = memory_commands.add_parser("list")
    list_command.add_argument("--user-id", required=True)
    list_command.add_argument("--category", choices=sorted(ALLOWED_CATEGORIES))
    list_command.add_argument("--limit", type=int, default=5)
    deactivate = memory_commands.add_parser("deactivate")
    deactivate.add_argument("--memory-id", type=int, required=True)
    return parser
```

- [ ] **Step 4: 组装命令执行路径**

把入口改为 `main(argv: list[str] | None = None)`，先解析 `args`，创建：

```python
repository = SQLAlchemyLongTermMemoryRepository(build_session_factory())
```

分支规则必须精确：

- `memory add` 调用 `.add()`，打印 `已新增长期记忆：<id>`；
- `memory list` 调用 `.list_active()`，逐行打印 `[<id>] <category>: <content>`，空结果打印 `无有效长期记忆。`；
- `memory deactivate` 调用 `.deactivate()`；真值打印成功，假值输出 stderr 并返回 `1`；
- `chat` 才创建 Retriever、Redis history 与聊天模型，然后用 `args.session_id`、`args.user_id` 和 Repository 调用 `ask_question()`；
- `ValueError`、MySQL 连接/查询异常打印 `错误：...` 到 stderr 并返回 `2`，不得调用模型。

- [ ] **Step 5: 追加成功解析测试并验证**

```python
def test_memory_commands_parse_expected_arguments():
    parser = build_parser()
    add = parser.parse_args([
        "memory", "add", "--user-id", "frank",
        "--category", "preference", "--content", "使用中文",
    ])
    deactivate = parser.parse_args(["memory", "deactivate", "--memory-id", "101"])
    assert (add.command, add.memory_command, add.user_id) == (
        "memory", "add", "frank"
    )
    assert deactivate.memory_id == 101
```

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_main.py -q
.venv\Scripts\python.exe -m py_compile main.py app\models.py app\database.py app\long_term_memory.py app\chat.py
```

Expected: 测试通过，`py_compile` 无输出。

### Task 6: 用 Alembic 生成并应用首个 MySQL 迁移

**Files:**
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/versions/<实际revision>_create_long_term_memories.py`

**Produces:** `alembic upgrade head` 在真实 MySQL 创建表和复合索引。

- [ ] **Step 1: 在私有 `.env` 填写非空 MySQL 配置并检查忽略规则**

```dotenv
MYSQL_DATABASE=agent_memory
MYSQL_USER=agent_app
MYSQL_PASSWORD=<本机私密应用密码>
MYSQL_ROOT_PASSWORD=<本机私密root密码>
MYSQL_URL=mysql+pymysql://agent_app:<URL编码密码>@localhost:3306/agent_memory
```

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git check-ignore -v .env
docker compose up -d mysql
docker compose ps
```

Expected: `.env` 被忽略，MySQL 健康后为 running。

- [ ] **Step 2: 初始化 Alembic**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\alembic.exe init migrations
```

在 `migrations/env.py` 添加：

```python
from app.database import get_mysql_url
from app.models import Base

config.set_main_option("sqlalchemy.url", get_mysql_url())
target_metadata = Base.metadata
```

保留生成的 offline/online 迁移函数；不把私密 URL 写进 `alembic.ini`。

- [ ] **Step 3: 生成、审查并应用迁移**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\alembic.exe revision --autogenerate -m "create long term memories"
.venv\Scripts\alembic.exe upgrade head
docker compose exec mysql mysql -u agent_app -p agent_memory -e "SHOW TABLES; SHOW INDEX FROM long_term_memories;"
```

MySQL 会交互要求应用账号密码，不要泄露。审查新迁移：`upgrade()` 创建表和 `ix_long_term_memories_user_active_category_updated`，`downgrade()` 对应删除它们。

### Task 7: 完整测试、真实验收、文档和学习者 Git 交付

**Files:**
- Modify: `README.md`
- Modify: `D:\桌面\所有codex项目\AI agent 开发\python 学习\README.md`
- Modify: 本计划勾选已完成项

- [ ] **Step 1: 运行完整离线测试**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: 全部 pytest 通过；现有 `RunnableWithMessageHistory` 弃用警告可保留；diff 检查无输出。

- [ ] **Step 2: 创建、查询、聊天与停用真实记忆**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe main.py memory add --user-id frank --category preference --content "回答时优先使用中文，并给出明确的 CMD 操作步骤。"
.venv\Scripts\python.exe main.py memory list --user-id frank
.venv\Scripts\python.exe main.py chat --session-id mysql-demo-a --user-id frank
```

记录新增的实际 ID。聊天后输入 `退出`，再运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe main.py chat --session-id mysql-demo-b --user-id frank
.venv\Scripts\python.exe main.py chat --session-id mysql-demo-c --user-id alice
.venv\Scripts\python.exe main.py memory deactivate --memory-id <实际ID>
docker compose stop mysql
.venv\Scripts\python.exe main.py chat --session-id mysql-demo-d --user-id frank
docker compose start mysql
```

Expected: 两个 `frank` 会话共用长期偏好、Redis 历史互不串；`alice` 不读到 `frank` 记忆；停用后不再读取；MySQL 停止时明确失败且不生成模型回答。

- [ ] **Step 3: 更新 README**

项目 README 必须说明：Redis/MySQL/RAG/qwen-plus 的职责、`session_id` 与 `user_id` 区别、Docker、迁移、CRUD CLI、最小权限账号、逻辑停用、索引、严格故障策略、完整测试和 Milvus 尚未实现。总 README 的“新窗口接续提示”必须记录真实测试数、提交哈希和下一节 Milvus（MySQL 仍为权威源）。

- [ ] **Step 4: 由学习者检查并精确提交**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git status --short
git check-ignore -v .env
git add requirements.txt compose.yaml .env.example alembic.ini migrations app\models.py app\database.py app\long_term_memory.py app\chat.py main.py tests\test_database.py tests\test_long_term_memory.py tests\test_chat.py tests\test_main.py README.md docs\superpowers\specs\2026-08-28-mysql-long-term-memory-design.md docs\superpowers\plans\2026-08-28-mysql-long-term-memory.md
git commit -m "feat: add mysql long-term memory"
git push origin main
git status -sb
```

Expected: 不包含 `.env` 或 `pytest-of-wurunnan/`；最后 `main...origin/main` 同步。总学习 README 不属于 Week07 仓库，不在此提交中。

## Plan Self-Review

- **Spec coverage:** Task 1 覆盖 Compose/私密配置；Task 2 覆盖模型和连接；Task 3 覆盖权威数据、隔离、筛选、停用；Task 4 覆盖 Prompt 和严格故障；Task 5 覆盖人工 CRUD CLI；Task 6 覆盖迁移；Task 7 覆盖测试、真实演示、文档和 Git。Milvus 明确留到下一节。
- **Placeholder scan:** 不含 TODO/TBD；动态迁移文件名明确按实际生成文件处理。
- **Type consistency:** `LongTermMemory` 被 Repository、渲染和测试共用；`.list_active(user_id, category=None, limit=5)` 在聊天与 CLI 使用同一签名；`ask_question()` 的 `user_id` 与 Repository 参数在测试和 CLI 中一致。

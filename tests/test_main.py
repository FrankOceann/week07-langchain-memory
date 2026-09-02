import pytest
from types import SimpleNamespace
from main import (
    build_parser,
    run_chat_command,
    run_memory_command,
    build_workflow_checkpointer,
)
import main as main_module
from app.workflow import build_minimal_graph

def test_chat_command_requires_session_id_and_user_id():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--session-id", "redis-demo"])

    args = parser.parse_args(
        ["chat", "--session-id", "redis-demo", "--user-id", "frank"]
    )

    assert (args.command, args.session_id, args.user_id) == (
        "chat",
        "redis-demo",
        "frank",
    )


def test_memory_add_requires_valid_category():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "memory",
                "add",
                "--user-id",
                "frank",
                "--category",
                "unknown",
                "--content",
                "内容",
            ]
        )

def test_memory_commands_parse_expected_arguments():
    parser = build_parser()

    add = parser.parse_args(
        [
            "memory",
            "add",
            "--user-id",
            "frank",
            "--category",
            "preference",
            "--content",
            "使用中文",
        ]
    )
    deactivate = parser.parse_args(
        ["memory", "deactivate", "--memory-id", "101"]
    )

    assert (add.command, add.memory_command, add.user_id) == (
        "memory",
        "add",
        "frank",
    )
    assert deactivate.memory_id == 101


def test_memory_outbox_drain_parses_limit():
    parser = build_parser()

    args = parser.parse_args(
        ["memory", "outbox", "drain", "--limit", "7"]
    )

    assert (
        args.command,
        args.memory_command,
        args.outbox_command,
        args.limit,
    ) == ("memory", "outbox", "drain", 7)


def test_memory_outbox_retry_failed_parses_all_flag():
    parser = build_parser()

    args = parser.parse_args(
        ["memory", "outbox", "retry-failed", "--all"]
    )

    assert (
        args.command,
        args.memory_command,
        args.outbox_command,
        args.all,
    ) == ("memory", "outbox", "retry-failed", True)


def test_outbox_drain_defers_milvus_connection_until_worker_runs(
    monkeypatch,
    capsys,
):
    class FakeOutboxWorker:
        def __init__(self, **kwargs):
            pass

        def drain(self, limit):
            assert limit == 10
            return {
                "succeeded": 0,
                "retrying": 0,
                "failed": 0,
            }

    def unavailable_milvus_client():
        raise AssertionError(
            "组装 Outbox worker 时不应连接 Milvus。"
        )

    monkeypatch.setattr(
        main_module,
        "build_session_factory",
        lambda: object(),
    )
    monkeypatch.setattr(
        main_module,
        "SQLAlchemyLongTermMemoryRepository",
        lambda session_factory: object(),
    )
    monkeypatch.setattr(
        main_module,
        "MemoryOutboxRepository",
        lambda session_factory: object(),
    )
    monkeypatch.setattr(
        main_module,
        "DashScopeEmbeddings",
        lambda: object(),
    )
    monkeypatch.setattr(
        main_module,
        "MemorySyncService",
        lambda embeddings, vector_index: object(),
    )
    monkeypatch.setattr(
        main_module,
        "MemoryOutboxWorker",
        FakeOutboxWorker,
    )
    monkeypatch.setattr(
        main_module,
        "build_milvus_client",
        unavailable_milvus_client,
    )

    exit_code = main_module.main(
        ["memory", "outbox", "drain"]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "索引任务处理结果：成功 0，重试 0，失败 0\n"
    )

def test_memory_outbox_drain_runs_worker(capsys):
    class RecordingOutboxWorker:
        def __init__(self):
            self.limits = []

        def drain(self, limit):
            self.limits.append(limit)
            return {
                "succeeded": 2,
                "retrying": 1,
                "failed": 0,
            }

    worker = RecordingOutboxWorker()
    args = SimpleNamespace(
        memory_command="outbox",
        outbox_command="drain",
        limit=7,
    )

    exit_code = run_memory_command(
        args,
        repository=object(),
        memory_sync_service=None,
        outbox_worker=worker,
    )

    assert exit_code == 0
    assert worker.limits == [7]
    assert capsys.readouterr().out == (
        "索引任务处理结果：成功 2，重试 1，失败 0\n"
    )


def test_memory_outbox_retry_failed_requeues_events(capsys):
    class RecordingOutboxRepository:
        def __init__(self):
            self.retry_times = []

        def retry_all_failed(self, now):
            self.retry_times.append(now)
            return 3

    outbox_repository = RecordingOutboxRepository()
    args = SimpleNamespace(
        memory_command="outbox",
        outbox_command="retry-failed",
        all=True,
    )

    exit_code = run_memory_command(
        args,
        repository=object(),
        memory_sync_service=None,
        outbox_repository=outbox_repository,
    )

    assert exit_code == 0
    assert len(outbox_repository.retry_times) == 1
    assert capsys.readouterr().out == (
        "已重新排队失败索引任务：3\n"
    )


def test_memory_add_creates_outbox_task_without_inline_sync(capsys):
    class SyncServiceThatMustNotRun:
        def sync(self, memory):
            raise AssertionError("memory add 不应同步 Milvus。")

    memory = SimpleNamespace(id=101, user_id="frank", content="优先使用中文")
    repository = FakeMemoryRepository(memory)
    args = SimpleNamespace(
        memory_command="add",
        user_id="frank",
        category="preference",
        content="优先使用中文",
    )

    exit_code = run_memory_command(
        args,
        repository,
        SyncServiceThatMustNotRun(),
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "已新增长期记忆并创建索引任务：101\n"
    )

class FakeMemoryRepository:
    def __init__(self, memory):
        self.memory = memory
        self.add_calls = []

    def add(self, user_id, category, content):
        self.add_calls.append(
            {
                "user_id": user_id,
                "category": category,
                "content": content,
            }
        )
        return self.memory


class RecordingMemorySyncService:
    def __init__(self):
        self.synced_memories = []

    def sync(self, memory):
        self.synced_memories.append(memory)


def test_memory_add_creates_outbox_task_without_syncing_milvus(capsys):
    memory = SimpleNamespace(id=101, user_id="frank", content="优先使用中文")
    repository = FakeMemoryRepository(memory)
    memory_sync_service = RecordingMemorySyncService()
    args = SimpleNamespace(
        memory_command="add",
        user_id="frank",
        category="preference",
        content="优先使用中文",
    )

    exit_code = run_memory_command(
        args,
        repository,
        memory_sync_service,
    )

    assert exit_code == 0
    assert repository.add_calls == [
        {
            "user_id": "frank",
            "category": "preference",
            "content": "优先使用中文",
        }
    ]
    assert memory_sync_service.synced_memories == []
    assert capsys.readouterr().out == (
        "已新增长期记忆并创建索引任务：101\n"
    )

def test_main_memory_add_does_not_create_sync_dependencies(
    monkeypatch,
    capsys,
):
    memory = SimpleNamespace(
        id=101,
        user_id="frank",
        content="优先使用中文",
    )
    vector = [0.1] * 1024
    captured = {}

    class FakeRepository:
        def __init__(self, session_factory):
            captured["repository"] = self

        def add(self, user_id, category, content):
            captured["add_arguments"] = {
                "user_id": user_id,
                "category": category,
                "content": content,
            }
            return memory

    class FakeEmbeddings:
        def embed_query(self, question):
            captured["embedded_question"] = question
            return vector

    class FakeVectorIndex:
        def __init__(self, client, collection_name):
            captured["vector_index"] = self
            captured["collection_name"] = collection_name
            self.upsert_calls = []

        def upsert(self, memory_id, user_id, vector):
            self.upsert_calls.append(
                {
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "vector": vector,
                }
            )

    monkeypatch.setattr(
        main_module,
        "build_session_factory",
        lambda: object(),
    )
    monkeypatch.setattr(
        main_module,
        "SQLAlchemyLongTermMemoryRepository",
        FakeRepository,
    )
    monkeypatch.setattr(
        main_module,
        "DashScopeEmbeddings",
        FakeEmbeddings,
    )
    monkeypatch.setattr(
        main_module,
        "build_milvus_client",
        lambda: object(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "MilvusMemoryVectorIndex",
        FakeVectorIndex,
        raising=False,
    )

    exit_code = main_module.main(
        [
            "memory",
            "add",
            "--user-id",
            "frank",
            "--category",
            "preference",
            "--content",
            "优先使用中文",
        ]
    )

    assert exit_code == 0
    assert captured["add_arguments"] == {
        "user_id": "frank",
        "category": "preference",
        "content": "优先使用中文",
    }
    assert "embedded_question" not in captured
    assert "collection_name" not in captured
    assert "vector_index" not in captured
    assert capsys.readouterr().out == (
        "已新增长期记忆并创建索引任务：101\n"
    )

class FailingMemorySyncService:
    def sync(self, memory):
        raise RuntimeError("Milvus 暂时不可用")


def test_memory_add_succeeds_when_sync_service_is_unavailable(capsys):
    memory = SimpleNamespace(
        id=101,
        user_id="frank",
        content="优先使用中文",
    )
    repository = FakeMemoryRepository(memory)
    memory_sync_service = FailingMemorySyncService()
    args = SimpleNamespace(
        memory_command="add",
        user_id="frank",
        category="preference",
        content="优先使用中文",
    )

    exit_code = run_memory_command(
        args,
        repository,
        memory_sync_service,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert repository.add_calls == [
        {
            "user_id": "frank",
            "category": "preference",
            "content": "优先使用中文",
        }
    ]
    assert captured.out == (
        "已新增长期记忆并创建索引任务：101\n"
    )
    assert captured.err == ""


def test_build_workflow_checkpointer_creates_sqlite_database(tmp_path):
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"

    checkpointer = build_workflow_checkpointer(checkpoint_path)

    try:
        checkpointer.setup()
        tables = {
            row[0]
            for row in checkpointer.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        checkpointer.conn.close()

    assert checkpoint_path.exists()
    assert {"checkpoints", "writes"} <= tables
def test_chat_command_uses_workflow_adapter(monkeypatch, capsys):
    args = SimpleNamespace(
        session_id="session-a",
        user_id="frank",
    )
    captured = {}
    questions = iter(["如何确认副作用操作？", "exit"])
    workflow_graph = object()
    checkpointer = object()

    class FakeEmbeddings:
        pass

    class FakeVectorIndex:
        def __init__(self, client, collection_name):
            self.client = client
            self.collection_name = collection_name

    class FakeSemanticMemoryService:
        def __init__(
            self,
            embeddings,
            vector_index,
            long_term_memory_repository,
        ):
            self.embeddings = embeddings
            self.vector_index = vector_index
            self.repository = long_term_memory_repository

    def fake_build_workflow(**kwargs):
        captured["graph_arguments"] = kwargs
        return workflow_graph

    def fake_ask_with_workflow(question, **kwargs):
        captured["question"] = question
        captured["adapter_arguments"] = kwargs
        return "图工作流回答", ["agent_safety.txt#chunk-0"]

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: next(questions),
    )
    monkeypatch.setattr(
        main_module,
        "DashScopeEmbeddings",
        FakeEmbeddings,
    )
    monkeypatch.setattr(
        main_module,
        "build_retriever",
        lambda data_directory, embeddings: object(),
    )
    monkeypatch.setattr(
        main_module,
        "build_redis_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        main_module,
        "RedisHistoryStore",
        lambda client, max_turns, ttl_seconds: object(),
    )
    monkeypatch.setattr(
        main_module,
        "build_chat_model",
        lambda: object(),
    )
    monkeypatch.setattr(
        main_module,
        "build_milvus_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        main_module,
        "MilvusMemoryVectorIndex",
        FakeVectorIndex,
    )
    monkeypatch.setattr(
        main_module,
        "SemanticLongTermMemoryService",
        FakeSemanticMemoryService,
    )
    monkeypatch.setattr(
        main_module,
        "build_workflow_checkpointer",
        lambda checkpoint_path: checkpointer,
    )
    monkeypatch.setattr(
        main_module,
        "build_chat_workflow_graph",
        fake_build_workflow,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "ask_question_with_workflow",
        fake_ask_with_workflow,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "build_conversation_runnable",
        lambda chat_model, history_store: (_ for _ in ()).throw(
            AssertionError("CLI 不应再创建旧对话 Runnable。")
        ),
        raising=False,
    )

    exit_code = run_chat_command(args, repository=object())

    assert exit_code == 0
    assert captured["question"] == "如何确认副作用操作？"
    assert captured["adapter_arguments"] == {
        "session_id": "session-a",
        "user_id": "frank",
        "workflow_graph": workflow_graph,
    }
    assert captured["graph_arguments"]["checkpointer"] is checkpointer
    assert capsys.readouterr().out == (
        "当前会话：session-a\n"
        "当前用户：frank\n"
        "输入 exit、quit 或 退出，结束对话。\n"
        "助手：图工作流回答\n"
        "=== agent_safety.txt#chunk-0 ===\n"
    )

def test_sqlite_checkpointer_restores_state_after_reopen(tmp_path):
    checkpoint_path = tmp_path / "workflow-checkpoints.sqlite"
    config = {"configurable": {"thread_id": "restart-test"}}

    first_checkpointer = build_workflow_checkpointer(checkpoint_path)
    try:
        first_graph = build_minimal_graph(first_checkpointer)
        first_graph.invoke(
            {"question": "第一轮问题"},
            config=config,
        )
    finally:
        first_checkpointer.conn.close()

    second_checkpointer = build_workflow_checkpointer(checkpoint_path)
    try:
        second_graph = build_minimal_graph(second_checkpointer)
        result = second_graph.invoke(
            {"question": "第二轮问题"},
            config=config,
        )
    finally:
        second_checkpointer.conn.close()

    assert result["completed_questions"] == [
        "第一轮问题",
        "第二轮问题",
    ]


def test_build_workflow_checkpointer_configures_wal_and_busy_timeout(tmp_path):
    checkpointer = build_workflow_checkpointer(
        tmp_path / "workflow-checkpoints.sqlite"
    )

    try:
        journal_mode = checkpointer.conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
        busy_timeout = checkpointer.conn.execute(
            "PRAGMA busy_timeout"
        ).fetchone()[0]
    finally:
        checkpointer.conn.close()

    assert journal_mode.lower() == "wal"
    assert busy_timeout == 5000


def test_chat_command_reports_workflow_error_and_continues(monkeypatch, capsys):
    args = SimpleNamespace(session_id="session-a", user_id="frank")
    questions = iter(["会失败的问题", "exit"])

    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))
    monkeypatch.setattr(main_module, "DashScopeEmbeddings", lambda: object())
    monkeypatch.setattr(main_module, "build_retriever", lambda *args: object())
    monkeypatch.setattr(main_module, "build_redis_client", lambda: object())
    monkeypatch.setattr(
        main_module, "RedisHistoryStore", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(main_module, "build_milvus_client", lambda: object())
    monkeypatch.setattr(
        main_module, "MilvusMemoryVectorIndex", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        main_module, "SemanticLongTermMemoryService", lambda **kwargs: object()
    )
    monkeypatch.setattr(main_module, "build_chat_model", lambda: object())
    monkeypatch.setattr(
        main_module, "build_workflow_checkpointer", lambda path: object()
    )
    monkeypatch.setattr(
        main_module, "build_chat_workflow_graph", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        main_module,
        "ask_question_with_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("ConnectionError: retriever unavailable")
        ),
    )

    assert run_chat_command(args, repository=object()) == 0
    assert "错误：ConnectionError: retriever unavailable" in (
        capsys.readouterr().out
    )

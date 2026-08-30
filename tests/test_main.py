import pytest
from types import SimpleNamespace
from main import (
    build_parser,
    run_chat_command,
    run_memory_command,
)
import main as main_module

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


def test_memory_add_syncs_saved_mysql_memory_to_milvus(capsys):
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
    assert memory_sync_service.synced_memories == [memory]
    assert capsys.readouterr().out == "已新增长期记忆：101\n"

def test_main_memory_add_syncs_mysql_memory_to_milvus(
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
    assert captured["embedded_question"] == "优先使用中文"
    assert captured["collection_name"] == "long_term_memory_vectors"
    assert captured["vector_index"].upsert_calls == [
        {
            "memory_id": 101,
            "user_id": "frank",
            "vector": vector,
        }
    ]
    assert capsys.readouterr().out == "已新增长期记忆：101\n"

class FailingMemorySyncService:
    def sync(self, memory):
        raise RuntimeError("Milvus 暂时不可用")


def test_memory_add_reports_sync_failure_after_mysql_is_saved(capsys):
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

    assert exit_code == 2
    assert repository.add_calls == [
        {
            "user_id": "frank",
            "category": "preference",
            "content": "优先使用中文",
        }
    ]
    assert captured.out == ""
    assert captured.err == (
        "长期记忆已写入 MySQL，但 Milvus 同步失败。请稍后重试同步。\n"
    )

def test_chat_command_passes_semantic_memory_service_to_question(
    monkeypatch,
):
    args = SimpleNamespace(
        session_id="session-a",
        user_id="frank",
    )
    repository = object()
    captured = {}
    questions = iter(["如何确认副作用操作？", "exit"])

    class FakeEmbeddings:
        pass

    class FakeVectorIndex:
        def __init__(self, client, collection_name):
            captured["vector_index"] = self
            captured["vector_index_client"] = client
            captured["collection_name"] = collection_name

    class FakeSemanticMemoryService:
        def __init__(
            self,
            embeddings,
            vector_index,
            long_term_memory_repository,
        ):
            captured["created_semantic_memory_service"] = self
            captured["semantic_embeddings"] = embeddings
            captured["semantic_vector_index"] = vector_index
            captured["semantic_repository"] = (
                long_term_memory_repository
            )

    def fake_ask_question(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return "假的回答", []

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
        "build_conversation_runnable",
        lambda chat_model, history_store: object(),
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
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "ask_question",
        fake_ask_question,
    )

    exit_code = run_chat_command(args, repository)

    assert exit_code == 0
    assert captured["question"] == "如何确认副作用操作？"
    assert captured["collection_name"] == "long_term_memory_vectors"
    assert captured["semantic_repository"] is repository
    assert (
        captured["semantic_vector_index"]
        is captured["vector_index"]
    )
    assert (
    captured["semantic_memory_service"]
    is captured["created_semantic_memory_service"]
)
# Week07 Milvus 语义长期记忆 Implementation Plan

**Goal:** 在 MySQL 保持唯一权威来源的前提下，以 Milvus 为已人工确认记忆建立可重试、幂等的语义候选索引。

**Architecture:** Milvus collection 只保存 `memory_id`、`user_id` 和 1024 维 embedding。`SemanticLongTermMemoryService` 先将问题向量化并在 Milvus 以 `user_id` 过滤检索候选 ID，再从 MySQL 回读、复核用户归属和启用状态，最后才把内容交给聊天提示词。写入顺序固定为 MySQL 提交成功后再 Milvus upsert；Milvus 从不成为事实来源。

**Tech Stack:** Python、现有 DashScope `text-embedding-v4`、LangChain Embeddings、SQLAlchemy、Milvus Standalone（etcd + MinIO + Milvus）、PyMilvus、Docker Compose、pytest。

**Spec:** 本会话中已确认的 Milvus 语义长期记忆架构（未另建设计文档，以学习者要求的“设计确认后直接写计划”顺序为准）。

## Global Constraints

- Windows CMD 命令一律先使用 `cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"`。
- 代码、测试、依赖、`.env`、Compose、README 和 Git 操作均由学习者亲自修改、执行；本文件只是操作清单。
- 不提交、发送或截图 `.env`、真实 API Key、MySQL 密码、Milvus 凭据、Docker 数据卷或 `pytest-of-wurunnan/`。
- MySQL `long_term_memories` 是唯一权威来源；Milvus 不保存 `content`、`category`、`source`、`is_active` 或审计字段。
- 模型没有新增、更新、停用或删除长期记忆的权限；只有人工 `memory` CLI 可触发这些操作。
- Milvus collection 使用 `text-embedding-v4` 默认的 **1024** 维向量。若未来更换模型或维度，必须创建新 collection 并从 MySQL 重新索引，不能混用向量。
- Redis 的 `session_id` 短期历史、MySQL 已有 CRUD/Alembic 迁移和本轮 RAG Retriever 保持原逻辑；仅为语义索引所必需的最小扩展可以修改 MySQL Repository。
- 本节不引入集群、高可用、鉴权、FastAPI、消息队列、Outbox、CDC 或异步 Worker。

---

## File Structure

- `requirements.txt`：增加 PyMilvus 客户端。
- `compose.yaml`：在保留 Redis/MySQL 的前提下增加 `etcd`、`minio`、`milvus` 和仅 Milvus 所需命名卷。
- `.env.example`：仅增加无敏感的 Milvus 地址、collection 名与向量维度示例；私有 `.env` 不提交。
- `app/milvus_memory.py`：Milvus collection 初始化、按 `memory_id` 幂等 upsert、按 `user_id` 搜索候选 ID；不含 MySQL、Prompt 或 CLI。
- `app/long_term_memory.py`：最小扩展为按指定候选 ID 回读同一用户的有效 MySQL 记录，并增加人工更新方法。
- `app/semantic_memory.py`：编排 Embedding、Milvus 和 MySQL；维护向量排名并执行最终过滤。
- `app/chat.py`：聊天前调用语义长期记忆 Service，而不是直接读取“最近 5 条”。
- `main.py`：装配 Milvus/语义 Service；将 `memory add`、新 `memory update` 与新 `memory sync` 接到“先 MySQL 后索引”的人工写入链路。
- `tests/test_milvus_memory.py`：离线 Fake Milvus 的 collection、upsert/search 契约测试。
- `tests/test_semantic_memory.py`：MySQL 最终过滤、排序、空结果、严格故障策略测试。
- `tests/test_long_term_memory.py`、`tests/test_chat.py`、`tests/test_main.py`：最小扩展既有行为回归测试。
- `README.md` 与总学习 `README.md`：只在全部验收后记录数据流、启动、演示和新的接续状态。

## Task 1: 先写语义检索的离线失败测试

**Files:**
- Create: `tests/test_semantic_memory.py`
- Create later: `app/semantic_memory.py`

**Why:** 先锁定“Milvus 只给候选 ID、MySQL 最终决定内容”的安全边界。此任务不需要 Docker、PyMilvus、`.env` 或网络。

**Interfaces to define in the test:**

```python
class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

class FakeVectorIndex:
    def upsert(self, memory_id: int, user_id: str, vector: list[float]) -> None: ...
    def search(self, user_id: str, vector: list[float], limit: int) -> list[int]: ...

class SemanticLongTermMemoryService:
    def search_active(self, user_id: str, question: str, limit: int = 3) -> list[LongTermMemory]: ...
    def sync(self, memory: LongTermMemory) -> None: ...
```

- [ ] **Step 1: 写第一个失败测试——同义问题按 Milvus 排名回读 MySQL。**

  Fake Vector Index 对 `frank` 返回 `[12, 10, 99]`；测试数据库只创建 `id=10`、`id=12` 的有效 `frank` 记忆。断言 `search_active("frank", "请按我的日常语言回复")` 的结果 ID 仍是 `[12, 10]`，而非 MySQL 的默认更新时间顺序。断言 Fake Embeddings 收到的是问题，Fake Index 同时收到 `user_id="frank"`、问题向量和 `limit=3`。

- [ ] **Step 2: 运行红灯。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest tests\test_semantic_memory.py -q
  ```

  Expected: 因 `app.semantic_memory` 尚不存在而失败。

- [ ] **Step 3: 追加最终过滤失败测试。**

  建立四种候选：其他用户 ID、已停用 ID、不存在 ID、有效当前用户 ID。Fake Index 返回全部四个 ID。断言最终只返回有效当前用户的记录；断言服务没有读取或渲染 Milvus 内容（Fake Index 仅允许返回整数 ID）。

- [ ] **Step 4: 追加严格失败测试。**

  分别使 `embed_query()`、`index.search()`、MySQL 候选回读抛出 `ConnectionError`。断言异常原样向上抛出，为聊天层的“模型零调用”测试提供边界。再增加空候选测试，断言返回空列表而非异常。

## Task 2: 实现 MySQL 候选回读与语义编排

**Files:**
- Modify: `app/long_term_memory.py`
- Create: `app/semantic_memory.py`
- Test: `tests/test_semantic_memory.py`

**Why:** MySQL 必须以 `user_id`、`is_active` 为条件再次验证 Milvus 返回的每个 ID；SQL 查询的自然返回顺序不能替代向量相似度排名。

- [ ] **Step 1: 在 `SQLAlchemyLongTermMemoryRepository` 新增候选回读方法。**

  新增签名：

  ```python
  def list_active_by_ids(self, user_id: str, memory_ids: list[int]) -> list[LongTermMemory]:
  ```

  对空 `memory_ids` 直接返回 `[]`。SQL 条件必须同时为 `LongTermMemory.id.in_(memory_ids)`、`LongTermMemory.user_id == user_id`、`LongTermMemory.is_active.is_(True)`。不要将 `user_id` 或 `is_active` 检查留给 Python；数据库查询本身也必须过滤。

- [ ] **Step 2: 实现 `app/semantic_memory.py` 的最小 Service。**

  构造函数接收三个依赖：`embeddings`、`vector_index`、`long_term_memory_repository`。`search_active()` 的精确顺序：`embed_query(question)` → `vector_index.search(user_id, vector, limit)` → `repository.list_active_by_ids(user_id, ids)` → 将回读记录构造成 `{memory.id: memory}` → 按原始 `ids` 顺序只保留存在的记录。不得调用现有 `.list_active()`，不得读取 Milvus 的文本字段。

- [ ] **Step 3: 运行 Task 1 的测试变绿。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest tests\test_semantic_memory.py -q
  ```

  Expected: 同义候选排序、跨用户/停用过滤、空结果和三类失败路径全部通过，且没有网络调用。

- [ ] **Step 4: 为候选回读加 Repository 回归测试。**

  在 `tests/test_long_term_memory.py` 以 SQLite 内存库创建两个用户、一个停用项和一个未知 ID；调用 `list_active_by_ids("frank", [...])`。断言仅返回 `frank` 的有效记录。运行：

  ```cmd
  .venv\Scripts\python.exe -m pytest tests\test_long_term_memory.py tests\test_semantic_memory.py -q
  ```

## Task 3: 先写 Milvus Adapter 的离线失败测试

**Files:**
- Create: `tests/test_milvus_memory.py`
- Create later: `app/milvus_memory.py`

**Why:** 将 PyMilvus 细节限制在一个 Adapter 内。测试只使用记录调用的 Fake Client，不启动 Milvus，也不要求 API Key。

- [ ] **Step 1: 写 collection schema/初始化测试。**

  目标类名为 `MilvusMemoryVectorIndex`。测试 Fake Client 时断言初始化 collection `long_term_memory_vectors` 的 schema 只有：`memory_id`（INT64 primary key）、`user_id`（VARCHAR scalar filter）、`embedding`（FLOAT_VECTOR, dim=1024）。断言创建向量索引使用 COSINE 度量；若 collection 已存在，不再次创建。

- [ ] **Step 2: 写幂等写入与用户过滤搜索测试。**

  调用两次 `upsert(memory_id=101, user_id="frank", vector=[...])`，断言两次传给客户端的主键都是 `101`，不生成新 ID。配置 Fake search 返回带 `memory_id` 的结果，断言 `search("frank", vector, limit=3)` 向客户端传入 `user_id == "frank"` 的过滤表达式并只返回整数 ID，且保持响应相似度排序。

- [ ] **Step 3: 写配置失败测试。**

  针对 `get_milvus_uri(uri="")` 抛出包含 `MILVUS_URI` 的 `ValueError`；针对构造参数 `dimension != 1024` 抛出说明 `text-embedding-v4` 默认维度不匹配的 `ValueError`。

- [ ] **Step 4: 运行红灯。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest tests\test_milvus_memory.py -q
  ```

  Expected: 因 `app.milvus_memory` 尚不存在而失败。

## Task 4: 增加 PyMilvus 与 Docker Compose Standalone

**Files:**
- Modify: `requirements.txt`
- Modify: `compose.yaml`
- Create or modify: `.env.example`

**Why:** 这是真实演示所需的基础设施；此前的所有语义/Adapter 测试依然不依赖它。

- [ ] **Step 1: 在 `requirements.txt` 末尾加入 `pymilvus`。**

  安装并验证导入：

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe -c "import pymilvus; print(pymilvus.__version__)"
  ```

  Expected: 输出版本号，无 `ModuleNotFoundError`。

- [ ] **Step 2: 在 `.env.example` 增加无敏感配置。**

  ```dotenv
  MILVUS_URI=http://localhost:19530
  MILVUS_COLLECTION=long_term_memory_vectors
  MILVUS_EMBEDDING_DIM=1024
  ```

  私有 `.env` 只复制值；不填写或展示真实 MySQL 密码/API Key。`MILVUS_EMBEDDING_DIM` 在本节只能是 `1024`。

- [ ] **Step 3: 在现有 `compose.yaml` 增加 Milvus Standalone 依赖。**

  使用 Milvus 官方 Standalone Compose 样例中的三个服务，固定同一兼容版本标签：

  - `etcd`：持久化至 `etcd_data`；暴露不必映射到宿主机。
  - `minio`：持久化至 `minio_data`；供 Milvus 使用其内部网络地址。
  - `milvus`：依赖 etcd 和 MinIO；映射 `19530:19530`；以 `standalone` 命令启动；使用 `milvus_data` 命名卷；加入健康检查。

  保留既有 `redis`、`mysql` 服务、端口和 `mysql_data` 卷，不调整它们的参数。不要把任何密码硬编码进 Compose；MinIO 示例凭据只用于本地学习容器，若样例支持环境变量则从私有 `.env` 注入。

- [ ] **Step 4: 只验证 Compose 结构。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  docker compose config
  git diff --check
  ```

  Expected: 同时显示 redis、mysql、etcd、minio、milvus；无 YAML 或 diff 格式错误。此时尚不运行容器。

## Task 5: 实现 Milvus Adapter，并让离线测试变绿

**Files:**
- Create: `app/milvus_memory.py`
- Test: `tests/test_milvus_memory.py`

**Why:** 只在此文件导入 `pymilvus`；其他业务模块依赖抽象的 `upsert/search` 协议，测试可替换为 Fake。

- [ ] **Step 1: 实现配置读取和构造边界。**

  定义 `get_milvus_uri(uri: str | None = None) -> str`，从参数或 `MILVUS_URI` 获取值，空值抛清晰 `ValueError`。构造函数接收 URI、collection 名、dimension、可选 client；明确拒绝非 1024 的 dimension。

- [ ] **Step 2: 实现 collection 初始化。**

  `ensure_collection()` 必须先检查 collection 是否存在；不存在时创建上述三字段 schema 和 COSINE 索引。每次真实 search/upsert 前可安全调用它。它必须可重复执行，不删除或重建已有 collection。

- [ ] **Step 3: 实现幂等 upsert 与过滤 search。**

  `upsert()` 先验证 `memory_id` 为正整数、`user_id.strip()` 非空、向量长度为 1024，再以 `memory_id` 作为 Milvus primary key 调用 upsert。`search()` 同样验证向量长度，使用 `user_id == <安全转义后的当前用户>` 的 Milvus filter、COSINE 搜索和调用方给出的 limit；只映射结果中的 `memory_id` 为 `list[int]`。任何客户端异常向上传递，不吞掉、不返回伪空结果。

- [ ] **Step 4: 验证 Adapter 离线测试。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest tests\test_milvus_memory.py -q
  ```

  Expected: 全部通过；不要求 Docker 服务运行。

## Task 6: TDD 实现人工新增、更新与补同步

**Files:**
- Modify: `app/long_term_memory.py`
- Modify: `app/semantic_memory.py`
- Modify: `main.py`
- Modify: `tests/test_long_term_memory.py`
- Modify: `tests/test_main.py`
- Create or extend: `tests/test_semantic_memory.py`

**Why:** 确保唯一的写入入口是人工 CLI，并且所有新建或更新都严格执行 MySQL 先提交、Milvus 后 upsert。失败不会回滚权威记录，重复补同步保持主键不变。

- [ ] **Step 1: 先写失败测试——Repository 更新。**

  在 SQLite 测试中调用：

  ```python
  repository.update(memory_id, category="preference", content="新的确认偏好")
  ```

  断言返回同一 ID、内容已更新、`updated_at` 更新；不存在或已停用的 ID 返回 `None`。不要允许模型调用此方法。

- [ ] **Step 2: 实现 `SQLAlchemyLongTermMemoryRepository.update()` 并运行该测试变绿。**

  在单一 Session 中用 `session.get()` 定位有效记录，校验 category/content，修改字段，commit、refresh 并返回该对象；不得新建第二条记录。

- [ ] **Step 3: 先写失败测试——同步顺序与失败保留 MySQL。**

  对 CLI 的 `memory add` 使用 Fake Repository、Fake Semantic Service 和记录调用顺序的 Fake。断言成功时顺序为 `repository.add` 后 `semantic_service.sync(memory)`；同步失败时仍可通过 Repository `list_active()` 查到刚新增的记忆，CLI stderr 包含“已保存到 MySQL，但尚未同步到 Milvus”，并返回非零状态。

- [ ] **Step 4: 扩展 argparse 与命令处理。**

  在 `memory` 子命令增加：

  ```text
  memory update --memory-id <int> --category <allowed> --content <text>
  memory sync --memory-id <int>
  ```

  `add`：先 Repository `.add()`，再 Service `.sync(memory)`；同步成功才打印完整成功。

  `update`：先 Repository `.update()`；返回 `None` 时失败；否则调用同一 `.sync(memory)`，以相同 `memory_id` 覆盖旧向量。

  `sync`：从 MySQL 精确读取该 ID 的有效记录（可新增 `get_active_by_id(memory_id)`），找不到则失败；找到后只执行 `.sync(memory)`。这是修复“已入 MySQL、未入 Milvus”的显式、幂等补同步入口。

- [ ] **Step 5: 为重复补同步写测试。**

  对同一 MySQL 记录执行两次 `memory sync`；断言 Fake Vector Index 收到两次相同 `memory_id`、相同 `user_id`，没有任何生成 ID 的接口被调用。

- [ ] **Step 6: 运行局部离线测试。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest tests\test_long_term_memory.py tests\test_main.py tests\test_semantic_memory.py -q
  ```

  Expected: 写入顺序、失败保留、更新同 ID、补同步幂等均通过，且不连真实 MySQL/Milvus。

## Task 7: 将聊天改接语义 Service 并验证模型零调用

**Files:**
- Modify: `app/chat.py`
- Modify: `main.py`
- Modify: `tests/test_chat.py`

**Why:** 聊天只接收已完成 Milvus 候选与 MySQL 复核的记忆；不得退回现有“最近 5 条”兜底策略。

- [ ] **Step 1: 写聊天失败测试。**

  将现有 `FakeLongTermMemoryRepository` 替换为 `FakeSemanticLongTermMemoryService`，其 `.search_active(user_id, question, limit=3)` 返回带 ID/category/content 的对象。断言 Prompt 只包含该返回值。

  再让 Fake Service 分别抛出 Embedding、Milvus、MySQL 错误；使用记录 `invoke()` 次数的 Fake Runnable，断言每种情形都是 0 次模型调用。

- [ ] **Step 2: 运行红灯。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest tests\test_chat.py -q
  ```

  Expected: 旧函数参数/调用方式不匹配而失败。

- [ ] **Step 3: 最小改动 `app/chat.py`。**

  将 `ask_question()` 的 `long_term_memory_repository` 依赖替换为 `semantic_long_term_memory_service`；在构建 Prompt 前调用 `.search_active(user_id, question, limit=3)` 并继续复用 `render_long_term_memories()`。Retriever、Redis history、Prompt 中“本轮 RAG 资料优先”的规则和来源输出不得改变。

- [ ] **Step 4: 在 `main.py` 的 chat 依赖组装中注入 Service。**

  仅 `chat` 命令创建 `DashScopeEmbeddings`、Milvus Adapter 与 Semantic Service；`memory add/update/sync` 也创建同步所需依赖。`memory list/deactivate` 不应要求 DashScope 或 Milvus 可用，因为它们是纯 MySQL 管理命令。

- [ ] **Step 5: 验证聊天测试变绿。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest tests\test_chat.py tests\test_main.py -q
  ```

  Expected: 现有同 session Redis 历史、RAG 来源和新语义长期记忆测试均通过；任何语义链路错误均模型零调用。

## Task 8: 全量离线回归、真实多轮演示和文档

**Files:**
- Modify: `README.md`
- Modify: `D:\桌面\所有codex项目\AI agent 开发\python 学习\README.md`

**Why:** 先以离线测试证明行为，再使用私有配置验证真实 Milvus；最后才记录结果。README 和 Git 均由学习者修改/执行。

- [ ] **Step 1: 完整离线测试。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe -m pytest -q
  git diff --check
  git status --short
  ```

  Expected: 全部测试通过；不包含 `.env` 或 `pytest-of-wurunnan/`。记录实际总测试数。

- [ ] **Step 2: 配置私有 `.env` 并启动真实服务。**

  确认私有 `.env` 有既有 DashScope/Redis/MySQL 变量及 `MILVUS_URI`、`MILVUS_COLLECTION`、`MILVUS_EMBEDDING_DIM=1024`，但不展示其敏感值。运行：

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  docker compose up -d
  docker compose ps
  .venv\Scripts\alembic.exe current
  ```

  Expected: Redis/MySQL/etcd/MinIO/Milvus 均可用，Alembic 仍为 `77ca49d48dd1`。

- [ ] **Step 3: 真实新增、同义召回与跨会话演示。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe main.py memory add --user-id frank --category preference --content "我更喜欢用中文分步骤说明。"
  .venv\Scripts\python.exe main.py chat --session-id milvus-demo-a --user-id frank
  ```

  在聊天中输入不直接包含“中文”的问题，例如“请按我平时习惯的语言给出清晰步骤”。记录命令输出中新增的真实 memory ID；退出后以新 `session_id` 重复问题，证明 Redis 历史独立而长期记忆仍可语义召回。

- [ ] **Step 4: 真实更新、停用与最终过滤演示。**

  用记录的 ID 执行 `memory update` 后再次聊天，验证相同 ID 的新内容生效。随后执行：

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  .venv\Scripts\python.exe main.py memory deactivate --memory-id <实际ID>
  .venv\Scripts\python.exe main.py chat --session-id milvus-demo-b --user-id frank
  ```

  Expected: 不手动删除 Milvus 向量；停用后该记忆仍不能进入 Prompt，证明 MySQL 最终过滤生效。

- [ ] **Step 5: 验证同步失败与补同步。**

  在不泄露配置的前提下暂时停止 `milvus` 服务，执行一次 `memory add`，记录 CLI 的“已保存到 MySQL，但尚未同步到 Milvus”及非零退出码；恢复服务后对该实际 ID 执行 `memory sync --memory-id <实际ID>` 两次。两次都应成功，且不新增 MySQL 行。演示结束后再次运行完整 pytest。

- [ ] **Step 6: 由学习者更新 README。**

  Week07 README 说明：MySQL/Milvus 权威边界、collection 字段、1024 维模型约束、读写数据流、`add/update/sync/deactivate` CLI、Compose 启动、严格故障策略、离线 Fake 测试和真实演示。总学习 README 的“新窗口接续提示”记录实际测试数、提交哈希、Milvus 已完成与下一周主题。

- [ ] **Step 7: 由学习者检查、暂存、提交和推送。**

  ```cmd
  cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
  git status --short
  git check-ignore -v .env
  git diff --check
  git add requirements.txt compose.yaml .env.example app\long_term_memory.py app\milvus_memory.py app\semantic_memory.py app\chat.py main.py tests\test_milvus_memory.py tests\test_semantic_memory.py tests\test_long_term_memory.py tests\test_chat.py tests\test_main.py README.md docs\superpowers\plans\2026-08-29-milvus-semantic-memory.md
  git commit -m "feat: add milvus semantic memory"
  git push origin main
  git status -sb
  ```

  Expected: 暂存区不含 `.env`、任何密钥、Docker volume 或 `pytest-of-wurunnan/`；`main` 与 `origin/main` 同步。

## Plan Self-Review

- **Spec coverage:** Tasks 1–2 覆盖语义候选、MySQL 权威过滤和排序；Tasks 3–5 覆盖 Milvus collection、幂等 upsert、用户过滤和 Compose；Task 6 覆盖 MySQL 先写及 add/update/retry；Task 7 覆盖聊天注入与模型零调用；Task 8 覆盖离线/真实验收、README 与学习者 Git 交付。
- **Scope check:** 队列、Outbox、CDC、鉴权、集群、高可用、FastAPI 均明确排除；本计划只交付单机 Compose 下可测试的派生向量索引。
- **Type consistency:** Adapter 始终只处理 `memory_id/user_id/vector`；Service 统一使用 `search_active()` 与 `sync()`；MySQL 负责 `LongTermMemory` 原始记录，聊天只消费 Service 的最终结果。

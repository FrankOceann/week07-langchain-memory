# Week07 LangChain Memory

> 当前可运行版本已完成 Redis 短期记忆、MySQL 长期记忆与 Milvus 语义召回。下方第一至四节保留每个阶段的学习记录；实际运行请优先使用本节的“当前系统运行手册”。

## 当前系统与运行手册

当前项目是一个本地 CLI 形式的 RAG + Memory 学习闭环。它将三种状态明确分开：

```text
用户问题
  -> Redis：按 session_id 读取最近 3 轮短期对话
  -> 本地 RAG Retriever：重新检索本轮资料 Top-3
  -> Milvus：按 user_id 做语义搜索，返回长期记忆候选 ID
  -> MySQL：验证候选记录的 user_id 与 is_active，并读取权威正文
  -> Chat 模型：使用资料作为事实依据、使用有效长期记忆作个性化参考
```

| 组件 | 边界 | 当前职责 |
| --- | --- | --- |
| Redis | `session_id` | 保存最近 3 个问答轮，30 分钟滑动 TTL。 |
| MySQL | `user_id`、`memory_id` | 长期记忆权威来源；负责内容、类别和软停用状态。 |
| Milvus | `user_id` 过滤 + `memory_id` | 只保存向量索引和候选 ID，不作为长期记忆的权威来源。 |
| 本地 RAG 资料 | `source` | 每轮回答的事实依据；默认检索 Top-3。 |

### 启动开发环境（Windows CMD）

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
copy .env.example .env
notepad .env
docker compose up -d
docker compose ps
.venv\Scripts\alembic.exe upgrade head
.venv\Scripts\python.exe -m pytest -q
```

私有 `.env` 除 DashScope、Redis 与 MySQL 配置外，还必须有：

```dotenv
MILVUS_URI=http://127.0.0.1:19530
```

`.env` 不可提交、发送或截图。Docker Compose 会启动 Redis、MySQL、etcd、MinIO 与 Milvus；Milvus 的 Python 客户端连接端口为 `19530`。

### 当前 CLI 用法（Windows CMD）

```cmd
.venv\Scripts\python.exe main.py memory add --user-id frank --category preference --content "回答时优先使用中文，并给出简洁要点。"
.venv\Scripts\python.exe main.py memory list --user-id frank
.venv\Scripts\python.exe main.py memory deactivate --memory-id 1

.venv\Scripts\python.exe main.py chat --session-id demo-session --user-id frank
```

当前完整离线测试基线为 **42 passed, 2 warnings**。两条 warning 来自 `RunnableWithMessageHistory` 的 LangChain 弃用提示；它将在后续 LangGraph 专题中迁移。离线测试不会调用真实 API Key，也不需要 Docker 服务。

## 第一节：LangChain Retriever

本项目用于学习如何使用 LangChain 复现 Week06 的手写 RAG Retriever。

当前目标：从本地 `.txt` 资料中检索最相关的 Top-3 文档块，并保留来源元数据。

## 本节完成了什么

```text
data/*.txt
  -> Document（文本块 + source 元数据）
  -> Embeddings（文本变为向量）
  -> InMemoryVectorStore（保存向量并按相似度检索）
  -> Retriever（默认返回 Top-3 Document）
```

命令行入口 `main.py` 接收问题、检查空输入、建立 Retriever，并打印每个检索结果的来源和正文。

## 组件职责

| 组件 | 来源 | 职责 |
| --- | --- | --- |
| `Document` | LangChain | 统一表示一段资料：`page_content` 存正文，`metadata["source"]` 存来源。 |
| `Embeddings` | LangChain | 定义“文本转向量”的标准接口：`embed_documents()` 和 `embed_query()`。 |
| `DashScopeEmbeddings` | 本项目 | 将 DashScope 的 OpenAI-compatible Embedding 接口适配为 LangChain `Embeddings`。 |
| `InMemoryVectorStore` | LangChain | 在内存中保存 Document 与向量，并按余弦相似度查找相近资料。 |
| Retriever | LangChain | `retriever.invoke(question)` 返回最相关的 `list[Document]`；本项目默认取 Top-3。 |
| `load_documents()` / `build_retriever()` | 本项目 | 分别负责受控读取与切分资料、组装向量库和 Retriever。 |

## 安装与离线测试（Windows CMD）

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
```

测试使用固定输出的 Fake Embeddings，不会调用 DashScope、不会消耗 API 额度，也不依赖真实 API Key。

## 第一节历史检索演示（Windows CMD）

> 本小节记录第一节完成时的独立 Retriever 入口；当前 `main.py` 已改为 `chat` 与 `memory` 子命令，实际运行请使用“当前系统与运行手册”。

先在本机创建私密配置：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
copy .env.example .env
notepad .env
```

在 `.env` 中只填写自己的 Key：

```dotenv
DASHSCOPE_API_KEY=你的真实密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

上面的 Base URL 对应中国（北京）的共享 DashScope 域名；如果 API Key 属于其他区域、试用域名或工作区专属域名，必须改成与该 Key 匹配的 OpenAI-compatible Base URL。

然后运行：

```cmd
.venv\Scripts\python.exe main.py "如何确认副作用操作？"
```

预期输出最多 3 组结果，每组有 `=== 文件名#chunk-N ===` 来源标题和对应正文。

> `.env` 已被 `.gitignore` 忽略：不要提交、不要发送、不要截图真实 API Key。

## 与 Week06 的关系

Week06 手写了资料切分、Embedding 调用、向量相似度排序和 Top-K 返回逻辑；Week07 没有改变 RAG 的基本原理，而是将其中的通用部分交给 LangChain 的标准抽象。

| 对比 | Week06 | Week07 |
| --- | --- | --- |
| 文档表示 | 自定义字典/数据结构 | LangChain `Document` |
| 文本转向量 | 项目代码直接调用 | 自定义适配器实现 LangChain `Embeddings` |
| 向量检索 | 项目代码实现 Top-K | `InMemoryVectorStore` + `as_retriever()` |
| 查询结果 | 可包含 `source`、`score`、`content` | 标准 Retriever 先返回 `Document`，不直接返回 score |
| 运行方式 | FastAPI 只读 API + Docker | 当前为本地 CLI 学习闭环 |

保留自己的 `load_documents()`、`DashScopeEmbeddings` 和 `build_retriever()`，是为了把业务规则（只读 `data/`、400 字符切分、50 字符重叠、来源格式、Top-3）固定在项目边界；使用 LangChain，是为了让内部组件遵循生态通用接口，未来可以替换向量库或接入链，而不需要重写业务规则。

## 第一节与第二节的历史边界

- 向量库只在内存中：程序退出后全部索引会消失。
- 本节不包含 FastAPI、Docker、数据库、持久化向量库、对话短期记忆或长期记忆。
- `numpy` 是 `InMemoryVectorStore` 进行余弦相似度计算所需的依赖，已写入 `requirements.txt`。

## 第二节：对话式 RAG 短期记忆

本节在 Retriever 之上加入本地、进程内的短期会话历史：同一 `session_id` 的后续问题可以引用最近问答，同时每一轮仍重新检索本地资料 Top-3。

```text
当前问题 + session_id
  -> Retriever.invoke()：本轮重新取得 Top-3 Document
  -> 当前资料正文 + 最近消息历史 + 当前问题
  -> RunnableWithMessageHistory
  -> DashScope ChatOpenAI（qwen-plus）
  -> 写入本轮 HumanMessage / AIMessage，并裁剪旧轮次
  -> 输出回答与本轮资料来源
```

### 短期记忆规则

- `SessionHistoryStore` 按 `session_id` 管理不同会话；不同 ID 不共享消息。
- 每个会话只保留最近 3 个完整问答轮（最多 6 条消息），避免上下文与成本无限增长。
- `RunnableWithMessageHistory` 在调用模型前读取该会话历史，在模型回答后写入本轮用户消息与 AI 消息。
- 记忆仅在当前 Python 进程中存在；退出 `main.py` 或重启程序后，全部历史清空。
- 检索不会因为有历史而跳过：每一轮都会重新返回本地资料 Top-3，并在 CLI 中打印对应 `source`。

### 安装与离线测试（Windows CMD）

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
```

当前离线基线为 **12 passed**。测试使用 Fake Retriever、Fake Chat Runnable 与确定性 Embeddings，不读取真实 API Key，不联网，也不消耗额度。测试覆盖会话隔离、历史裁剪、同会话追问会带入前文、来源保留，以及缺少 `DASHSCOPE_API_KEY` 时的安全失败。

### 历史多轮演示（Windows CMD）

> 本小节的命令对应第二节完成时的入口形式。当前会话入口为 `main.py chat --session-id <ID> --user-id <用户>`，请使用文档顶部的当前 CLI 用法。

私有 `.env` 应包含与区域匹配的 `DASHSCOPE_API_KEY` 与 `DASHSCOPE_BASE_URL`；它已被 `.gitignore` 忽略，绝不能提交、发送或截图真实 Key。

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe main.py
```

默认会话 ID 为 `demo-session`。在同一进程中依次输入问题和追问，例如“如何确认副作用操作？”、“那为什么？”。输入 `exit`、`quit` 或 `退出` 结束对话。也可以把会话 ID 作为可选参数传入：

```cmd
.venv\Scripts\python.exe main.py session-a
```

### 当前限制

- 使用 `RunnableWithMessageHistory` 是本节的学习目标；当前 LangChain 会发出弃用警告，后续学习 LangGraph 持久化时再比较迁移方案。
- 系统提示词要求模型只把本轮检索资料当作事实依据，历史只用于理解“那为什么？”等指代；但真实模型仍可能在文字中提及先前对话资料。因此当前实现展示的是基础约束，不是严格的来源验证或生产级 grounding 保证。
- 第二节的进程内实现不包含 Redis、数据库、长期记忆、FastAPI、工具调用或持久化向量库；下一节已将相同的 `session_id` 会话边界迁移至带 TTL 的 Redis。

## 第三节：Redis 会话短期记忆

本节将短期记忆从 Python 的 `dict` 迁移至 Redis。`RunnableWithMessageHistory`、Retriever 和提示词结构保持不变；变化的是消息历史后端：历史按 `session_id` 写入 Redis List，因此 Python 进程退出后，只要键尚未过期，下一次启动仍可读取。

```text
main.py
  -> RedisHistoryStore.get(session_id)
  -> RedisChatMessageHistory
  -> Redis List: week07:chat_history:{session_id}
  -> 最近消息 + 本轮检索资料
  -> RunnableWithMessageHistory + ChatOpenAI
  -> 写入本轮用户消息和 AI 消息，裁剪并刷新 TTL
```

### Redis 存储规则

- 每个会话使用独立键，例如 `week07:chat_history:redis-demo`；不同 `session_id` 不共享消息。
- `RedisChatMessageHistory` 把每条 LangChain `HumanMessage` / `AIMessage` 序列化为 JSON，作为 Redis List 的一个元素保存；读取时再还原为原始消息类型。
- 每次写入依次执行 `RPUSH`、`LTRIM`、`EXPIRE`：追加本轮消息、保留最近 3 轮（最多 6 条）、将 TTL 重置为 1800 秒（30 分钟）。
- 这属于滑动 TTL：持续聊天会刷新 30 分钟；停止聊天后，Redis 自动删除整个会话键。
- `clear()` 只删除当前会话键，不影响其他会话。

### 本机 Redis 服务与私有配置（Windows CMD）

`compose.yaml` 只启动 Redis，不容器化 Python 应用。首次使用需启动 Docker Desktop，然后运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
docker compose up -d redis
docker compose exec redis redis-cli PING
```

预期最后输出 `PONG`。私有 `.env` 需要额外包含：

```dotenv
REDIS_URL=redis://localhost:6379/0
```

`.env.example` 只提供无凭据示例；真实 `.env` 仍被 `.gitignore` 忽略，绝不能提交、发送或截图。

停止或恢复本机 Redis：

```cmd
docker compose stop redis
docker compose start redis
```

### 离线测试与真实验收

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest -q
```

当前离线基线为 **15 passed, 1 warning**。新增测试通过 `fakeredis` 验证 Redis 会话隔离、最近轮次裁剪、TTL、`clear()` 和缺少 `REDIS_URL` 的失败路径；测试不需要 Docker、网络或真实 API Key。

真实验收已完成：Docker Compose Redis 返回 `PONG`；`redis-demo` 完成两轮问答后，`TTL` 返回正数且 `LLEN` 为 4；重启 Python 并使用相同 `session_id` 后，追问“上一轮的‘那’指什么？”仍能正确引用之前的主题；手动把演示键设为 1 秒 TTL 后，Redis 返回 `EXISTS = 0`，确认自动过期删除。

### 当前限制

- Redis 只提供短期会话状态，不是长期语义记忆、关系数据库或向量数据库。下一节才会复用 Week05 SQLModel Memory API 学习长期记忆。
- 本机 Compose 是单 Redis 容器学习环境，不包含认证、TLS、集群、高可用、监控或生产部署；当前未声明 Docker volume，因此执行 `docker compose down` 并删除容器后，容器内 Redis 数据不会作为持久化数据保留。
- Redis 不可用时，客户端会抛出明确的 `redis.exceptions.ConnectionError`，并且不会静默回退为进程内历史；更友好的 CLI 错误展示留作后续工程化改进。
- `RunnableWithMessageHistory` 仍会发出 LangChain 弃用警告。本节保留它以专注学习状态后端迁移，LangGraph 持久化将在后续专题比较。

## 第四节：MySQL 结构化长期记忆

本节在 Redis 短期会话状态之外，引入 MySQL 作为已确认长期记忆的权威数据源。它解决的是“同一用户跨会话、跨进程仍可保留的偏好、资料和事实”；它不是聊天记录的替代品。

```text
当前问题 + session_id + user_id
  -> Redis：按 session_id 读取最近 3 轮会话消息（30 分钟滑动 TTL）
  -> Retriever：为本轮问题重新检索 Top-3 资料
  -> MySQL：按 user_id 查询有效长期记忆
  -> 提示词：长期记忆仅作个性化参考，本轮检索资料仍是事实依据
  -> ChatOpenAI（qwen-plus）生成回答
```

### 三类状态的边界

| 组件 | 主键/边界 | 保存内容 | 生命周期与职责 |
| --- | --- | --- | --- |
| Redis | `session_id` | 最近用户/助手消息 | 短期会话上下文；最多 3 轮，30 分钟滑动 TTL。 |
| MySQL | `user_id` | 人工确认的 `preference`、`profile`、`fact` | 长期结构化记忆；可跨会话读取，是权威数据源。 |
| RAG 资料 | `source` | 本地资料块 | 本轮知识事实来源；每轮重新检索 Top-3。 |

`user_id` 与 `session_id` 不能混用：一个用户可以创建多个会话；同一个会话也只属于一个当前用户。不同用户的 MySQL 记忆绝不能互相注入提示词。

### 数据模型与安全规则

表 `long_term_memories` 包含 `id`、`user_id`、`category`、`content`、`source`、`is_active`、`created_at`、`updated_at`，并建立 `(user_id, is_active, category, updated_at)` 复合索引，以支持“某用户的有效记忆”查询。

- `category` 仅允许 `preference`、`profile`、`fact`。
- 删除采用软停用：`is_active=False`；数据不被物理删除，保留审计线索。
- 写入只通过明确的 `memory add` CLI 进行；模型不会自行新增、修改或停用记忆。
- MySQL 查询发生在聊天模型调用之前；数据库不可用时，CLI 输出明确错误且不调用模型生成回答。
- 长期记忆只用于个性化参考，不能覆盖本轮 RAG 资料，也不应被当作新的知识事实来源。

### 本机 MySQL、迁移与私有配置（Windows CMD）

当前 `compose.yaml` 定义 Redis、MySQL、etcd、MinIO 与 Milvus。本项目的 MySQL 容器将宿主机 `13306` 映射到容器内 `3306`，避免占用常见的本机 `3306` 端口。

私有 `.env` 除已有 DashScope 与 Redis 配置外，还需要填写 `MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_ROOT_PASSWORD` 和与实际端口一致的 `MYSQL_URL`。不要提交、发送或截图该文件或任何真实密码。

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
docker compose up -d mysql
docker compose ps
.venv\Scripts\alembic.exe upgrade head
.venv\Scripts\alembic.exe current
```

本节使用 Alembic 管理 schema。初始迁移位于 `migrations/versions/77ca49d48dd1_create_long_term_memories.py`；`alembic_version` 表记录数据库已经执行到的迁移版本。不要手工在生产环境改表后跳过迁移文件。

### 长期记忆 CLI（Windows CMD）

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"

.venv\Scripts\python.exe main.py memory add --user-id frank --category preference --content "回答时优先使用中文，并给出明确的 CMD 操作步骤。"
.venv\Scripts\python.exe main.py memory list --user-id frank
.venv\Scripts\python.exe main.py memory deactivate --memory-id 1

.venv\Scripts\python.exe main.py chat --session-id mysql-demo-a --user-id frank
```

最后一条命令进入对话。对话读取的是 `frank` 的有效长期记忆，而 Redis 消息历史仍由 `mysql-demo-a` 这个会话 ID 隔离。

### 验证结果与当前边界

离线完整测试基线为 **25 passed, 2 warnings**。测试覆盖 MySQL URL/模型定义、Repository 用户隔离与软停用、聊天提示词只读取当前用户的记忆，以及“长期记忆查询失败时模型零调用”的失败路径。测试不会连接真实 MySQL、Redis 或调用真实模型。

真实验收已验证：Docker MySQL 健康检查通过；Alembic 创建 `long_term_memories` 与 `alembic_version`；添加、列出、停用记忆可用；同一用户跨 `session_id` 能读取偏好，不同用户不能读取该偏好；停止 MySQL 后聊天明确失败且没有生成助手回答，恢复服务后可继续使用。

- `RunnableWithMessageHistory` 仍存在 LangChain 弃用警告；本项目保留它以继续学习消息历史后端与状态边界，后续再比较 LangGraph。
- 当前检索在注入历史之前以原始问题执行。因此“那为什么？”这类强依赖上下文的追问可能先检索不到足够资料并返回“资料不足”。这是严格 RAG 来源约束下的已知限制；后续可专门学习“历史感知的查询改写”。
- MySQL 是结构化长期记忆的权威源；第五节已接入 Milvus 作为语义候选索引，并在应用层回读和过滤 MySQL 的有效记录。

## 第五节：Milvus 语义长期记忆

本节在 MySQL 权威长期记忆之上引入 Milvus。Milvus 的职责不是保存完整业务记录，而是把记忆内容转换为向量后，按照“意思是否相近”快速召回候选 `memory_id`；候选记录必须回 MySQL 验证后才能进入模型上下文。

```text
写入：memory add
  -> MySQL 保存 LongTermMemory，生成 memory_id
  -> DashScope Embeddings 生成 1024 维向量
  -> Milvus upsert：memory_id + user_id + embedding

读取：chat
  -> 当前问题生成 1024 维向量
  -> Milvus 按 user_id 过滤并返回相近 memory_id
  -> MySQL 按 user_id、memory_id、is_active=True 回读
  -> 按 Milvus 相似度顺序注入有效长期记忆
```

### 数据模型与安全边界

Milvus collection 名称为 `long_term_memory_vectors`，包含以下字段：

| 字段 | 用途 |
| --- | --- |
| `memory_id` | MySQL `long_term_memories.id`；Milvus 主键与关联键。 |
| `user_id` | Milvus 搜索过滤条件，避免跨用户候选召回。 |
| `embedding` | DashScope 文本 embedding，维度固定为 1024，使用 COSINE 距离。 |

- MySQL 是长期记忆的权威源；Milvus 只保存可重建的索引数据。
- 聊天读取时必须再次校验 `user_id` 和 `is_active=True`，因此 MySQL 软停用能立即阻止旧 Milvus 向量被使用。
- 模型不能自行写入、修改或停用长期记忆；写入仅经显式 `memory add` CLI。
- Milvus、MySQL 或 embedding 任一环节异常时，不静默回退，也不调用聊天模型生成回答。

### 本机验收（Windows CMD）

启动全部依赖并完成迁移后，可以按以下顺序验证：

```cmd
.venv\Scripts\python.exe main.py memory add --user-id frank --category preference --content "用户 frank 偏好中文、简洁回答。"
.venv\Scripts\python.exe main.py chat --session-id milvus-demo --user-id frank
```

在聊天中输入“之后请用什么风格回答我？”，应能得到与这条长期偏好一致的回答。再换用新的 `session_id`，同一 `user_id` 仍能得到该偏好，证明它来自跨会话长期记忆而不是 Redis 历史。

也应使用另一用户（如 `bob`）进行相同提问，确认不会召回 Frank 的偏好；停用某条记忆后，即使直接查询 Milvus 仍能看到该向量，聊天也不应再使用它。

### Milvus 一致性说明

Milvus 的默认 bounded staleness 一致性可能造成“刚写入、立即搜索尚未可见”的短暂现象。本项目已在手动验收中观察到这一点。生产场景不能简单地在每次写入后强制 flush；应按业务场景明确选择一致性策略：例如“写后立即展示”的交互可考虑 Session 或 Strong，而可容忍短暂不可见的批量检索可使用 Bounded。

### 当前限制

- 当前写入同步发生在 CLI 进程内：MySQL 成功但 Milvus 失败时仅提示稍后重试，尚无后台补偿机制。
- `user_id` 目前由 CLI 参数提供，尚未接入真实认证、授权和租户身份。
- Milvus 为本机 Standalone 学习部署，未包含 TLS、RBAC、备份、高可用、监控或容量规划。
- 当前 RAG 文档仍使用进程内 `InMemoryVectorStore`；Milvus 目前只索引长期记忆，而不是 `data/` 中的 RAG 资料。

## 面向企业开发的后续学习路线

学习不以“继续叠加框架”为目标，而以每一阶段解决一个真实工程风险为目标。

| 优先级 | 学习主题 | 要解决的企业问题 | 交付物 |
| --- | --- | --- | --- |
| 1 | Outbox 异步索引同步 | MySQL 成功、Milvus 失败时，索引可能永久缺失。 | `memory_outbox`、幂等 worker、重试/失败状态、补偿 CLI 与测试。 |
| 2 | 历史感知查询改写与 RAG 评测 | “那为什么？”等追问用原始问题检索，召回质量不足。 | 查询改写边界、固定评测集、Recall@K/来源正确性记录。 |
| 3 | FastAPI + 身份认证 + 租户授权 | CLI 参数不能代表可信身份；企业不能相信客户端传来的 user_id。 | JWT 身份解析、服务端 tenant/user 边界、接口测试。 |
| 4 | 可观测性与真实集成测试 | 出现慢请求、同步堆积或串租户时，需要可定位、可报警。 | 结构化日志、请求 ID、耗时/失败指标、Docker 集成测试。 |
| 5 | LangGraph 状态迁移 | `RunnableWithMessageHistory` 已弃用，需要可恢复、可审计的工作流状态。 | 迁移设计、checkpointer、状态回放与回归测试。 |
| 6 | 部署、安全与运维 | 学习环境不等于生产环境。 | 应用容器化、密钥管理、TLS/RBAC、备份恢复、健康检查与发布流程。 |

下一阶段从 **Outbox 异步索引同步** 开始：它直接补上当前架构最重要的数据一致性风险，并训练事务、异步任务、幂等、重试和可观测性这些企业高频能力。

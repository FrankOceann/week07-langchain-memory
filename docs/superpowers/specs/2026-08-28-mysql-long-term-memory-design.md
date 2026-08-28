# Week07 MySQL 结构化长期记忆设计

**日期：** 2026-08-28
**状态：** 已确认，待实施计划

## 目标

在 Week07 项目中从零加入 MySQL 结构化长期记忆。Redis 继续保存按 `session_id` 隔离、会过期的短期会话历史；MySQL 保存按 `user_id` 隔离、不会因 Redis TTL 删除的已确认长期记忆。对话只读取长期记忆，模型不能自动写入或修改长期记忆。

本节不复用 Week05 项目。Week07 自己通过 Docker Compose 启动 MySQL，并以 SQLAlchemy、Alembic 和 Repository 层管理持久化数据。

## 范围

- 在现有 `compose.yaml` 中增加 MySQL 8.4 服务、数据卷与健康检查，保留 Redis 服务。
- 通过私有 `.env` 的 `MYSQL_URL` 连接 MySQL；容器初始化凭据也只保存在私有 `.env`。
- 使用 SQLAlchemy 定义 `long_term_memories` 表，并使用 Alembic 迁移创建和演进表结构。
- 实现 Repository 层，提供新增、按用户查询有效记忆和逻辑停用。
- CLI 提供人工 `memory add`、`memory list`、`memory deactivate` 命令；聊天命令只读长期记忆。
- 聊天时组合 Redis 历史、MySQL 长期记忆、本轮 RAG 资料和当前问题，继续使用已有 `qwen-plus` 聊天模型。
- 离线测试不依赖真实 MySQL；真实验收使用 Docker Compose 中的 MySQL。

不包含自动从聊天提炼记忆、模型自动写库、FastAPI、认证授权、多租户权限服务、队列、Outbox、MySQL 高可用/读写分离/分库分表或 Milvus 实现。Milvus 是下一节：它只为本表中的已确认记忆建立语义索引。

## 分层职责

```text
Redis
  短期会话消息；按 session_id；最近 3 轮；30 分钟滑动 TTL。

MySQL
  长期记忆权威来源；按 user_id；结构化、可筛选、可逻辑停用。

RAG Retriever
  本轮知识事实与来源。

qwen-plus
  阅读受控上下文并生成回答；不直接读写数据库。
```

`session_id` 和 `user_id` 必须独立。一个用户可同时拥有多个 Redis 会话；这些会话读取同一用户的长期记忆，但不会共享聊天历史。

## 数据模型

表名：`long_term_memories`。

| 字段 | 建议类型 | 规则与用途 |
| --- | --- | --- |
| `id` | `BIGINT` | 自增主键；未来 Milvus 通过它关联权威记录。 |
| `user_id` | `VARCHAR(64)` | 非空；记忆归属用户。 |
| `category` | `VARCHAR(32)` | 非空；初始允许 `preference`、`profile`、`fact`。 |
| `content` | `TEXT` | 非空；已确认的长期记忆正文。 |
| `source` | `VARCHAR(32)` | 非空；初始由人工 CLI 写入 `user_confirmed`。 |
| `is_active` | `BOOLEAN` | 非空，默认 `true`；停用而非物理删除。 |
| `created_at` | `DATETIME` | 非空；创建时间，UTC。 |
| `updated_at` | `DATETIME` | 非空；新增、修改和停用时更新，UTC。 |

建立复合索引：`(user_id, is_active, category, updated_at)`。常见读取路径是：查询一个用户、只取有效记忆、可选按分类过滤、按最近更新时间限制条数。

不对 `content` 加唯一约束：是否重复是业务判断，不能仅凭文本相等阻断用户明确保存的记录。

## CLI 与写入边界

聊天命令使用显式身份参数：

```text
python main.py chat --session-id redis-demo --user-id frank
```

人工管理命令：

```text
python main.py memory add --user-id frank --category preference --content "..."
python main.py memory list --user-id frank
python main.py memory deactivate --memory-id 101
```

`memory deactivate` 只将目标记录的 `is_active` 更新为 `false`，不执行物理删除。所有写入来自明确的人工 CLI 命令；对话生成模型不得调用 Repository 的写入方法。

## 一次聊天的数据流

```text
chat --session-id redis-demo --user-id frank
  -> Redis：读取 redis-demo 的短期消息历史
  -> MySQL：读取 frank 的有效长期记忆
  -> Retriever：检索本轮 RAG 资料与来源
  -> Prompt：系统规则 + 历史 + 长期记忆 + 本轮资料 + 当前问题
  -> qwen-plus：生成回答
  -> Redis：写入本轮用户消息和 AI 消息
```

Prompt 的上下文权限必须明确：

1. 系统规则优先级最高，禁止模型自动新增或修改长期记忆。
2. 本轮 RAG 资料用于知识事实和引用。
3. MySQL 长期记忆用于已确认的用户偏好、资料和事实；不能覆盖本轮 RAG 资料。
4. Redis 历史只用于理解指代，不能作为新的事实依据。

`qwen-plus` 已由 `app/chat.py` 的 `ChatOpenAI(model="qwen-plus")` 使用。`text-embedding-v4` 是 RAG 向量模型，不是聊天模型；本节不新增模型服务。

## 故障策略

- 缺少 `MYSQL_URL`：启动或需要数据库的命令立即报清晰配置错误。
- MySQL 无法连接或查询失败：明确提示长期记忆服务不可用，本轮不调用聊天模型；不静默跳过长期记忆。
- 查询结果为空：正常继续对话，并向 Prompt 提供“无已确认长期记忆”。
- `memory deactivate` 找不到记录或记录已停用：命令返回清晰的未找到/无可变更状态，而不是伪造成功。

本节采取严格模式，便于验证和排错。后续 FastAPI 工程化可增加“回答降级但带状态标记”的可配置模式。

## 数据库配置与迁移

私有 `.env` 将保存 MySQL 容器初始化变量和 `MYSQL_URL`。`.env.example` 只保留空值或无敏感示例，真实密码不得提交、发送或截图。

Python 应用使用最小权限的 MySQL 应用账号，不能使用 root 账号。MySQL 的数据保存在命名 Docker volume `mysql_data`，容器重建后不会因容器文件系统删除而丢失。

Alembic 是唯一的 schema 演进路径：

```text
SQLAlchemy 模型变更
  -> 新建并审查 Alembic migration
  -> alembic upgrade head
  -> 数据库达到确定版本
```

应用启动不会隐式创建表；修改表结构必须新增迁移，不能修改已经应用过的旧迁移。

## 组件边界

- `app/models.py`：SQLAlchemy Declarative Base 和 `LongTermMemory` 表模型。
- `app/database.py`：从 `MYSQL_URL` 创建 Engine 和 Session 工厂；缺配置时抛出清晰错误。
- `app/long_term_memory.py`：Repository 接口与 MySQL 实现；不包含提示词、模型调用或 CLI 解析。
- `app/chat.py`：将只读长期记忆渲染为独立 Prompt 上下文；保留 RAG 来源行为。
- `main.py`：使用 `argparse` 区分 `chat` 与 `memory` 子命令，创建依赖并将参数传给对应层。

## 测试与验收

### 离线 pytest

使用 Fake Repository 和 Fake Chat Model，不连接 MySQL 或真实模型，验证：

1. `user_id` 隔离：用户 A 读取不到用户 B 的记忆。
2. 查询只返回 `is_active=true` 的记录，并能按类别、更新时间和数量限制读取。
3. 停用只影响目标 `memory_id`。
4. Prompt 能接收长期记忆，同时保留本轮 RAG 来源和 Redis 历史行为。
5. 长期记忆查询失败时不会调用聊天模型。
6. 缺少 `MYSQL_URL` 时数据库工厂失败且信息清晰。
7. 现有 Redis、Retriever 和对话测试继续通过。

### 真实演示

1. 使用 Docker Compose 启动 Redis 与 MySQL，确认各自健康。
2. 配置私有 `.env`，运行 Alembic migration。
3. 用 `memory add` 为 `frank` 添加至少一条 `preference`。
4. 启动 `chat --session-id ... --user-id frank`，验证回答遵循该偏好。
5. 换一个 `session_id` 但保持 `user_id=frank`，验证仍读取同一长期记忆、短期历史不串会话。
6. 换成另一 `user_id`，验证不读取 `frank` 的记忆。
7. 停用记忆后再次聊天，验证该偏好不再进入 Prompt。
8. 停止 MySQL，验证应用明确失败而非静默回答。

## 后续 Milvus 接口

Milvus 在下一小节实现，关系固定为：

```text
MySQL long_term_memories
  id=101, user_id=frank, content=...
     -> embedding
Milvus memory_vectors
  memory_id=101, user_id=frank, vector=...
```

语义检索先从 Milvus 得到候选 `memory_id`，再回 MySQL 按 `id`、`user_id` 和 `is_active=true` 获取最终正文。Milvus 不保存或判定权威状态，不允许跳过 MySQL 的用户隔离和启用状态检查。

首版 Milvus 将使用显式同步/重建索引命令；Outbox、消息队列、异步重试和双写一致性是后续工程化主题。

## 已知限制

- 当前 `RunnableWithMessageHistory` 有 LangChain 弃用警告；本节不迁移 LangGraph，避免将数据库学习与工作流框架迁移混在一起。
- 本节是单机 Docker Compose 演示，不覆盖 MySQL 高可用、备份恢复演练、数据库权限平台、Milvus 集群或生产监控。
- 长期记忆是结构化候选上下文，不是模型永久“学会”的参数更新。

# Week07 Outbox 异步索引同步设计

**日期：** 2026-08-30  
**状态：** 已确认，待实施计划

## 目标

将长期记忆写入从“MySQL 提交后在 CLI 进程中直接调用 Embedding 和 Milvus”改为可靠的 Outbox 模式。MySQL 继续是唯一权威来源；Milvus 是可重建的派生索引。只要权威记忆事务提交成功，对应的索引任务也必须持久化，之后可重试、可观察、可人工补偿。

## 范围与非范围

本节交付 MySQL Outbox、幂等的手动 worker CLI、有限重试与失败状态、人工重放失败任务，以及离线 pytest 覆盖。保留现有 Redis、RAG、MySQL/Milvus 用户隔离与 MySQL 最终校验。

本节不引入 Celery、Redis Queue、Kafka、常驻进程、定时调度器、FastAPI、认证、分布式追踪或生产级监控。`memory outbox drain` 是显式运行的学习型 worker。

## 一致性模型

```text
memory add / deactivate
  -> 一个 MySQL 事务：更新 long_term_memories + 插入 memory_outbox
  -> 提交成功后：权威数据与待处理索引任务同时可见
  -> memory outbox drain：认领事件，回读 MySQL 权威记录
       -> active：Embedding + Milvus upsert(memory_id 为主键)
       -> inactive / 不存在：删除 Milvus memory_id
       -> 成功：succeeded；异常：重试或 failed
```

Outbox 事件不保存长期记忆正文或向量，只保存 `memory_id` 和索引请求元数据。worker 每次都回读 MySQL：这避免旧事件携带过时内容，也使“旧 upsert 事件在停用之后才执行”安全地变成删除操作。

MySQL 提交与 Milvus 写入仍不是跨库原子事务；Outbox 的承诺是 **at-least-once 投递**。重复投递由 Milvus `memory_id` 主键 upsert 和幂等删除吸收，最终达到索引与权威状态一致。

## 数据模型

新增表 `memory_outbox`：

| 字段 | 用途 |
| --- | --- |
| `id` | BIGINT 主键，事件标识。 |
| `memory_id` | `long_term_memories.id`；worker 的唯一业务目标。 |
| `event_type` | 固定 `memory.index_requested`，表达“按当前权威状态调和索引”。 |
| `status` | `pending`、`processing`、`succeeded`、`failed`。 |
| `attempt_count` | 已执行次数。 |
| `available_at` | 下一次可认领时间，实现退避。 |
| `lease_token`、`lease_expires_at` | worker 认领令牌与过期租约，恢复崩溃遗留的 processing 任务。 |
| `last_error` | 最后一次失败的截断错误摘要，不保存密钥。 |
| `processed_at` | 成功完成时间。 |
| `created_at`、`updated_at` | UTC 审计时间。 |

建立 `(status, available_at, id)` 索引供 worker 扫描。事件一经创建不删除；失败事件保留以便人工诊断和重放。

## 写入和执行边界

`SQLAlchemyLongTermMemoryRepository.add()` 与 `.deactivate()` 各自在其原有 Session 内插入对应 Outbox 行并一次 `commit()`。因此数据库写入失败时两者一起回滚；数据库提交成功时事件一定存在。

`MemoryOutboxWorker` 通过 Outbox Repository 认领一条到期 `pending` 事件或租约过期的 `processing` 事件，写入新 token 和有限期租约。它回读 `memory_id`：有效记录则调用既有 `MemorySyncService`，无效/停用记录则调用 Milvus delete。只有 token 匹配时才能更新成功或失败状态，避免两个 worker 覆盖对方结果。

每次失败增加 `attempt_count`。未达到 `max_attempts` 时写回 `pending` 并按固定、可测试的退避规则推迟 `available_at`；达到上限时标为 `failed` 并保留错误。异常不吞没为“成功”。

## CLI

```text
python main.py memory outbox drain --limit 10
python main.py memory outbox retry-failed --all
```

`drain` 只处理至多 `limit` 条已经到期的事件，并报告成功、重试、失败计数。它可被人工反复运行，且不要求常驻后台服务。

`retry-failed --all` 是人工补偿入口：将所有 `failed` 事件恢复为 `pending`、清空错误和租约、设置为立即可运行。它只重放已有的持久化任务，不直接绕过 Outbox 访问 Milvus。

`memory add` 成功后只报告 MySQL 记忆和 Outbox 事件已创建；不再因 Milvus 临时不可用返回失败。`memory deactivate` 同理创建调和事件。聊天继续按现有 MySQL `user_id`、`is_active` 最终校验，因而停用在索引删除重试期间也立即安全。

## 测试策略

全部使用 SQLite 内存库、Fake Embeddings 与 Fake Milvus Adapter，不读取 `.env`、不启动 Docker：

1. 新增和停用时，长期记忆与 Outbox 事件在同一事务提交。
2. MySQL 写入成功后，即使 worker/Embedding/Milvus 失败，事件仍是可重试的 `pending` 或 `failed`，而非丢失。
3. 同一事件或同一 `memory_id` 被重复处理，Milvus 接收同一主键 upsert/delete，不产生新权威记录。
4. 首次失败进入退避重试；达到上限进入 `failed`；`retry-failed --all` 后可重新认领并成功。
5. 过期租约能被重新认领；未过期的 processing 事件不能被另一 worker 窃取。
6. 停用记忆的 worker 执行删除；即使删除失败，聊天的 MySQL 最终过滤仍不使用该记忆。
7. CLI 参数、输出和依赖装配测试不要求 DashScope 或 Milvus 服务。

## 文件边界

- `app/models.py`：`MemoryOutbox` ORM 模型及状态常量。
- `migrations/versions/<revision>_create_memory_outbox.py`：唯一 schema 变更路径。
- `app/outbox.py`：Outbox Repository、认领/状态转换和 worker；不解析 CLI，也不渲染 Prompt。
- `app/long_term_memory.py`：在同一 MySQL 事务内写记忆和 Outbox 事件。
- `app/memory_sync.py`：保留 Embedding + Milvus upsert；新增/使用 delete 由 worker 调和。
- `app/milvus_memory.py`：增加幂等的 `delete(memory_id)` Adapter 操作。
- `main.py`：添加 `memory outbox` 命令并移除 `memory add` 的同步依赖。
- `tests/`：按模型、Repository、worker、Milvus Adapter 和 CLI 分层测试。
- `README.md`：更新写入时序、命令、限制与真实验收步骤。

## 约束

- `.env` 中的真实密钥、密码和连接串不得读取、展示或修改。
- `pytest-of-wurunnan/` 是 pytest 临时目录，不提交、不删除。
- 所有面向学习者的命令优先提供 Windows CMD 版本。
- MySQL 始终是长期记忆内容、归属与启用状态的唯一权威来源；Milvus 永远只返回候选 `memory_id`。

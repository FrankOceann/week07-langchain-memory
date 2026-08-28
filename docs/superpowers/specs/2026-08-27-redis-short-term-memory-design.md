# Week07 Redis 会话短期记忆设计

**日期：** 2026-08-27
**状态：** 已确认，待实施计划

## 目标

将第二节中仅存在于 Python 进程内的会话消息历史迁移到 Redis。保留已有对话式 RAG 的行为：同一 `session_id` 可引用最近问答、不同会话互相隔离、每轮仍重新检索资料。应用重启后，只要 Redis 中的会话尚未过期，历史仍可被读取。

## 范围

- 使用 Docker Compose 在本机启动单个 Redis 服务。
- Python 从私有 `.env` 的 `REDIS_URL` 读取连接地址。
- 用 Redis List 保存 LangChain 消息，并实现兼容 `BaseChatMessageHistory` 的 Redis 适配器。
- 对每个会话保留最近 3 轮，即最多 6 条用户/AI 消息。
- 每次写入后刷新该会话键的 TTL，采用滑动过期。
- 离线测试使用 `fakeredis`，真实演示使用 Docker Compose 启动的 Redis。

不包含长期记忆、SQLModel 数据库、FastAPI、Redis 集群/高可用/认证/监控、持久化向量库或 LangGraph 迁移。

## 架构与数据流

```text
main.py
  -> ask_question()
  -> RunnableWithMessageHistory
  -> RedisChatMessageHistory(session_id)
  -> redis-py client
  -> Docker Compose 中的 Redis
```

1. CLI 获得 `session_id` 和本轮问题。
2. Retriever 仍在本轮检索资料；这部分不迁移到 Redis。
3. `RunnableWithMessageHistory` 通过历史工厂取得该 `session_id` 的 `RedisChatMessageHistory`。
4. 适配器从 Redis List 读取最近消息，供模型理解指代关系。
5. 模型生成回答后，适配器把本轮用户消息和 AI 消息追加到同一 Redis List。
6. 适配器裁剪旧消息，并刷新该键 TTL。
7. 用户停止对话后，TTL 到期，Redis 自动删除该会话历史。

## Redis 键与存储格式

- 键格式：`week07:chat_history:{session_id}`，例如 `week07:chat_history:demo-session`。
- 键类型：Redis List；列表从左到右按消息发生顺序排列。
- 单条元素：使用 LangChain 的消息序列化格式转换为 JSON 字符串，保留消息类型和内容，避免把 AI、用户消息混为普通文本。
- 追加：使用 `RPUSH` 保持时间顺序。
- 裁剪：使用 `LTRIM` 保留末尾 `max_turns * 2` 条消息；默认 `max_turns=3`，即最多 6 条。
- 删除：`clear()` 删除当前会话键，不影响其他 `session_id`。

## TTL 规则

- 默认 TTL：1800 秒（30 分钟），作为短期会话状态。
- 每次成功写入消息后调用 `EXPIRE`，将 TTL 重置为完整 1800 秒；这是滑动过期。
- 只读取历史不会刷新 TTL，避免无实际对话的会话永久存在。
- 过期后 Redis 自动删除整个 List；下一次同一 `session_id` 视为新会话。
- `TTL` 用于真实演示和测试中检查键是否确实带过期时间。

## 组件职责

### `RedisChatMessageHistory`

新增到 `app/memory.py`，实现 `BaseChatMessageHistory`：

- `messages`：从 Redis List 读取 JSON，反序列化为 LangChain `BaseMessage` 列表。
- `add_messages(messages)`：序列化、追加、裁剪、刷新 TTL；四个 Redis 操作通过 pipeline 一起提交。
- `clear()`：删除当前会话键。

它只负责消息持久化，不负责模型调用、检索、提示词或 CLI 输入。

### Redis 客户端工厂

新增一个明确的创建函数：从传入的 URL 或 `REDIS_URL` 环境变量创建 `redis.Redis` 客户端。缺少 URL 时抛出清晰异常；真实服务不可用时让连接错误清晰暴露，不静默退回进程内字典，防止“以为已持久化、实际没有”的错误。

### 对话装配

`app/chat.py` 和 `main.py` 只替换历史工厂的来源：由原来的 `SessionHistoryStore.get` 改为为指定会话创建 Redis 历史对象。Retriever、提示词、`RunnableWithMessageHistory` 的输入键和输出键保持不变。

## 本地服务与配置边界

- 新增 `compose.yaml`，仅定义 Redis 服务和本机端口映射；不会打包或容器化 Python 应用。
- 私有 `.env` 增加 `REDIS_URL=redis://localhost:6379/0`；`.env` 继续由 `.gitignore` 忽略。
- `.env.example` 只提供无凭据示例地址和配置说明。
- 生产环境替换 `REDIS_URL` 为企业 Redis/托管 Redis 地址；代码不写死主机、端口或密码。

## 测试与验收

### 离线 pytest

使用 `fakeredis` 构建测试客户端，验证：

1. 不同 `session_id` 读取不到彼此消息。
2. 超过 3 轮时只保留最后 6 条消息。
3. 写入后 `TTL` 为正数；手动短 TTL 到期后该会话为空。
4. `clear()` 仅删除目标会话。
5. 缺少 `REDIS_URL` 时客户端创建失败且报错清晰。
6. 已有对话 RAG 测试仍能证明同一会话可使用前一轮消息、来源仍按本轮 Retriever 返回。

### 真实演示

1. 用 Docker Compose 启动 Redis。
2. 配置本机私有 `.env`，运行交互式 CLI 并完成至少两轮问答。
3. 退出并重新运行 Python CLI，使用相同 `session_id` 提问，确认历史仍被带入。
4. 用 `redis-cli TTL` 或等价检查确认键有正数 TTL；使用短 TTL 演示到期后历史消失。
5. 关闭 Redis 后验证应用给出明确连接错误。

## 已知限制

- 仍使用 `RunnableWithMessageHistory`，当前 LangChain 会提示其弃用；本节保持它，以便只学习状态后端迁移。LangGraph 持久化迁移留给后续专题。
- Redis 是短期状态，不是长期语义记忆；TTL 到期后数据会删除。
- Redis List 的容量按消息条数而非 token 精确裁剪；更精细的 token 控制不在本节范围。

# Week07 第二节：对话式 RAG 短期记忆设计

## 目标

在已有 LangChain Retriever 之上实现一个本地命令行对话式 RAG：同一 `session_id` 的后续问题可以引用最近的问答；每次回答仍只以本地 `data/` 检索出的 Top-3 文档为事实依据。

## 范围与非目标

- 本节使用内存保存消息，进程退出后历史清空。
- 使用 DashScope 的 OpenAI-compatible 接口调用通义聊天模型生成回答。
- 不加入 FastAPI、Redis、数据库、长期记忆、工具调用、Docker 或持久化向量库。
- Redis 留到下一小节：把已验证的会话历史迁移到带 TTL 的进程外存储。

## 架构与职责

| 组件 | 职责 |
| --- | --- |
| `app/documents.py` | 受控读取、切分资料，并保留 `source` 元数据。 |
| `app/embeddings.py` | 将文本转为向量，供 Retriever 使用。 |
| `app/retriever.py` | 建立内存向量库，返回相关 Top-3 `Document`。 |
| `app/chat.py` | 创建聊天模型、构造带资料和历史的提示词，并提供对话 Runnable。 |
| `app/memory.py` | 按 `session_id` 提供独立的内存消息历史，并限制保留的最近轮数。 |
| `main.py` | 校验命令行输入，调用对话 Runnable，输出回答和资料来源。 |

## 数据流

```text
question + session_id
  -> Retriever.invoke(question) 获取 Top-3 Document
  -> 提取 source 与正文，作为本轮 context
  -> RunnableWithMessageHistory 读取该 session 的历史消息
  -> Prompt(系统规则 + 历史 + context + 当前问题)
  -> DashScope Chat Model 生成回答
  -> RunnableWithMessageHistory 写入本轮 HumanMessage / AIMessage
  -> CLI 输出回答与来源
```

`session_id` 是会话隔离键：不同值绝不能共享消息。CLI 第一版允许通过可选参数传入；未传入时使用明确的默认演示会话名。

## 提示词与上下文边界

系统提示词要求模型仅依据本轮检索资料回答；资料没有覆盖时必须明确说明“资料不足”，不得把历史消息当作新的事实来源。历史用于理解指代和追问，例如“那为什么？”中的“那”。每个会话只保留最近固定数量的完整问答轮，防止 token 和成本持续增长；裁剪在写入/读取会话历史的边界完成。

## 错误处理

- 空白问题复用现有 `validate_question()`，返回错误退出码。
- 缺少 `DASHSCOPE_API_KEY` 在创建真实聊天模型前给出清晰 `ValueError`。
- 无可索引资料沿用 Retriever 的 `ValueError`。
- 聊天模型请求失败不打印密钥、请求头或内部堆栈。

## 测试策略

测试通过注入 Fake Chat Model 和确定性 Embeddings 完成，不读取 `.env`、不联网、不会产生 API 费用。至少覆盖：会话 A 的前文会进入其第二轮提示词、会话 A/B 严格隔离、超出上限时旧消息被裁剪、资料来源被保留、空问题与缺少配置被拒绝。真实演示才使用私有 `.env` 和 DashScope。

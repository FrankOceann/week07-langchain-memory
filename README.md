# Week07 LangChain Memory

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

## 真实检索演示（Windows CMD）

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

## 当前边界

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

### 真实多轮演示（Windows CMD）

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
- 本节不包含 Redis、数据库、长期记忆、FastAPI、Docker、工具调用或持久化向量库。下一节才会将相同的 `session_id` 会话边界迁移至带 TTL 的 Redis。

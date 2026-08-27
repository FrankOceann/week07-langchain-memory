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

# Week07：LangChain Retriever 设计

## 目标

在独立的 `week07-langchain-memory` 项目中，用 LangChain 复现 Week06 已手写完成的资料检索能力。第一节只完成 Retriever 最小闭环：本地 `.txt` 资料、固定切分、向量化、Top-3 检索和来源元数据。

Week06 保持不变，作为手写 RAG API 的可运行对照项目。

## 选择

采用 `langchain-core` 的下列组件：

- `Document`：保存 Chunk 正文与来源元数据。
- `Embeddings`：将已有 DashScope Embedding 调用适配为 LangChain 接口。
- `InMemoryVectorStore`：在内存中保存 Chunk 向量并进行相似度检索。
- `VectorStore.as_retriever()`：取得标准 Retriever，并通过 `invoke(question)` 检索资料。

不在本节使用 Chroma、FAISS、数据库或云端向量库。它们适合后续持久化与规模化练习，但会分散本节对 LangChain 基础接口的学习重点。

## 项目边界

第一节计划形成如下结构：

```text
week07-langchain-memory/
├── app/
│   ├── documents.py       # 读取 .txt 并切分为 Document
│   ├── embeddings.py      # DashScope 的 LangChain Embeddings 适配器
│   └── retriever.py       # 建立向量库并返回 Retriever
├── data/                  # 演示资料
├── tests/
│   └── test_retriever.py  # 离线 Retriever 测试
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

每个模块只承担一个职责：

1. `documents.py` 将受限 `data/` 目录下的 `.txt` 文件切分为 `Document`。
2. `embeddings.py` 将文本或问题转换为向量。
3. `retriever.py` 将 Document 与 Embeddings 放入 `InMemoryVectorStore`，并返回 Top-3 Retriever。

## 数据流

建立索引：

```text
data/*.txt
→ 固定大小、带重叠的 Chunk
→ Document(page_content, metadata)
→ Embeddings.embed_documents()
→ InMemoryVectorStore
→ Retriever
```

执行检索：

```text
用户问题
→ retriever.invoke(question)
→ Embeddings.embed_query()
→ InMemoryVectorStore 相似度检索
→ 前 3 个 Document
```

每个 Chunk 的 `metadata["source"]` 固定为 `文件名#chunk-N`。这样保留 Week06 的来源引用能力。

标准 `retriever.invoke()` 返回 `list[Document]`，而不是 Week06 API 的 `source`、`score`、`content` JSON。第一节验证正文、来源与顺序；分数包装留给后续需要 API 输出时再设计。

## 运行与错误边界

- 真实运行从 `.env` 读取 DashScope API Key；Key 不进入代码、测试或 Git。
- 没有 API Key 时，真实构建或检索应给出明确错误，不发起无效网络请求。
- `InMemoryVectorStore` 只在进程内存中保存向量；程序结束后索引消失，下次启动会重新向量化资料。这是本节明确的非持久化边界。
- 没有可索引资料时，建立 Retriever 的函数抛出清晰的 `ValueError`。
- 命令行演示在调用 `retriever.invoke()` 前拒绝空白问题；标准 Retriever 本身不额外包裹输入校验。

## 测试策略

测试不使用真实 DashScope：

1. 使用 Fake Embeddings 为固定文本和问题返回确定向量。
2. 建立最小临时资料目录和 Retriever。
3. 断言 Top-3 数量、相关性排序、`page_content` 与 `metadata["source"]`。
4. 覆盖空资料的构建失败，以及命令行演示的空白问题拒绝等边界。

因此 `pytest` 离线、稳定、不消耗 API 额度。

## 非目标

本节不实现或修改以下内容：

- Week06 项目的任何代码、测试或 Docker 配置；
- FastAPI、Docker、LLM 回答生成链；
- 短期记忆、长期记忆、SQLModel Memory API；
- Chroma、FAISS、数据库、持久化向量库；
- LangGraph、MCP、认证、上传或索引重建。

## 完成标准

- `pytest` 通过，且 Retriever 测试均离线运行。
- `retriever.invoke(question)` 能返回最多 3 个相关 `Document`，保留 `source` 元数据。
- 配置真实环境变量后可完成一次 DashScope 检索演示。
- README 说明安装、测试、真实演示、组件映射与内存索引边界。

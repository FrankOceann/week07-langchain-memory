# LangChain Retriever 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立 Week07 项目中，以 LangChain `InMemoryVectorStore` 实现可离线测试、可真实演示的 Top-3 Retriever。

**Architecture:** `app/documents.py` 只负责把受限 `.txt` 资料转成带来源元数据的 LangChain `Document`。`app/embeddings.py` 只负责将 DashScope 的 OpenAI-compatible Embedding 调用适配为 LangChain `Embeddings`。`app/retriever.py` 使用 `InMemoryVectorStore` 建库并通过 `as_retriever()` 返回标准 Retriever；`main.py` 只负责读取命令行问题并打印结果。

**Tech Stack:** Python 3.10+, `langchain-core`, OpenAI Python SDK（DashScope OpenAI-compatible endpoint）, `python-dotenv`, `pytest`。

**Spec:** `docs/superpowers/specs/2026-08-27-langchain-retriever-design.md`

## Global Constraints

- Week07 是独立 Git 项目；不得修改或移动 Week06 的文件。
- 只新增 `langchain-core`；本节不引入 FastAPI、Docker、Chroma、FAISS、数据库、LangGraph 或 MCP。
- 只读取项目内 `data/` 目录直接包含的 UTF-8 `.txt` 文件。
- Chunk 规则固定为 400 字符、50 字符重叠；来源格式固定为 `文件名#chunk-N`。
- 默认检索 Top-K 为 3；标准 `retriever.invoke()` 只返回 `list[Document]`，不包装 `score`。
- 测试必须使用 Fake Embeddings，不能访问网络、读取 `.env` 或消耗 DashScope API 额度。
- API Key 只能来自本地 `.env`，不得提交到 Git。
- 对用户展示的命令必须适用于 Windows CMD，并以 `cd /d` 切换目录。

---

## 文件结构与职责

| 文件 | 职责 |
| --- | --- |
| `.gitignore` | 忽略本机虚拟环境、缓存和 `.env`。 |
| `.env.example` | 只给出 DashScope 变量名与公开 Base URL。 |
| `requirements.txt` | 声明运行、测试与 LangChain Core 依赖。 |
| `app/documents.py` | 从资料目录加载并切分 `Document`。 |
| `app/embeddings.py` | 实现 `DashScopeEmbeddings(Embeddings)`。 |
| `app/retriever.py` | 以 `InMemoryVectorStore` 创建 `VectorStoreRetriever`。 |
| `main.py` | 验证命令行问题、建立 Retriever、打印检索结果。 |
| `tests/test_retriever.py` | 对资料切分、Fake Embeddings、Retriever 及边界进行离线测试。 |
| `README.md` | 记录安装、测试、真实演示、组件映射与内存边界。 |

### Task 1: 初始化独立项目与可运行的最小骨架

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `data/agent_safety.txt`
- Create: `data/rag_long_test.txt`
- Create: `tests/__init__.py`
- Create: `README.md`
- Create: `docs/superpowers/specs/2026-08-27-langchain-retriever-design.md` (already exists)
- Create: `docs/superpowers/plans/2026-08-27-langchain-retriever.md` (already exists)

**Interfaces:**
- Consumes: 已确认的设计文档。
- Produces: 一个不含密钥、可安装依赖的独立 Git 项目目录；后续任务使用 `app`、`data`、`tests` 目录。

- [ ] **Step 1: 初始化 Git，且只检查新 Week07 目录**

在 Windows CMD 中运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git init
git branch -M main
git status --short
```

预期：只显示 Week07 自己的未跟踪 `docs/` 目录；不得出现 Week06 内容。

- [ ] **Step 2: 创建最小配置与目录文件**

创建 `.gitignore`：

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
```

创建 `.env.example`：

```dotenv
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

创建 `requirements.txt`：

```text
openai
python-dotenv
pytest
langchain-core
```

创建空文件 `app/__init__.py` 与 `tests/__init__.py`。创建简短 `README.md`，包含标题 `# Week07 LangChain Memory` 与“第一节：LangChain Retriever”的说明。

- [ ] **Step 3: 复制公开演示资料，不复制 `.env`、虚拟环境或 Week06 源代码**

在 Windows CMD 中运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
copy "..\week06-llm-tool-calling\data\agent_safety.txt" "data\agent_safety.txt"
copy "..\week06-llm-tool-calling\data\rag_long_test.txt" "data\rag_long_test.txt"
dir data
```

预期：`data` 中仅出现两份公开 `.txt` 演示资料。不要复制 Week06 的 `.env`、`.venv`、`.git`、`.worktrees`、Dockerfile 或测试备份文件。

- [ ] **Step 4: 创建并安装 Week07 虚拟环境**

在 Windows CMD 中运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "from langchain_core.documents import Document; from langchain_core.vectorstores import InMemoryVectorStore; print('langchain-core ready')"
```

预期：最后一行输出 `langchain-core ready`。若 `py -3.10` 不可用，先运行 `py -0p`，然后选用已安装的 Python 3.10+ 路径。

- [ ] **Step 5: 验证忽略规则并首次提交项目骨架与设计文档**

在 Windows CMD 中运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git status --short
git add .gitignore .env.example requirements.txt app\__init__.py tests\__init__.py data\agent_safety.txt data\rag_long_test.txt README.md docs\superpowers\specs\2026-08-27-langchain-retriever-design.md docs\superpowers\plans\2026-08-27-langchain-retriever.md
git commit -m "chore: initialize week07 langchain retriever project"
git status --short
```

预期：提交后工作区干净；`.env` 不应出现在 `git status --short` 中。

### Task 2: 先用 TDD 定义 Document 切分与来源格式

**Files:**
- Create: `tests/test_retriever.py`
- Create: `app/documents.py`

**Interfaces:**
- Consumes: `data_directory: pathlib.Path`，以及固定 `CHUNK_SIZE = 400`、`CHUNK_OVERLAP = 50`。
- Produces: `load_documents(data_directory: Path) -> list[Document]`；每个 Document 必有 `page_content` 与 `metadata["source"]`。

- [ ] **Step 1: 写失败测试，先描述切分和来源契约**

在 `tests/test_retriever.py` 写入：

```python
from app.documents import load_documents


def test_load_documents_preserves_chunk_source_and_overlap(tmp_path):
    text = "A" * 400 + "B" * 100
    (tmp_path / "guide.txt").write_text(text, encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.metadata["source"] for document in documents] == [
        "guide.txt#chunk-0",
        "guide.txt#chunk-1",
    ]
    assert documents[0].page_content == "A" * 400
    assert documents[1].page_content == "A" * 50 + "B" * 100
```

- [ ] **Step 2: 运行测试，确认它因模块不存在而失败**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py -q
```

预期：FAIL，错误为 `ModuleNotFoundError: No module named 'app.documents'`。

- [ ] **Step 3: 写最小实现，只加载直接 `.txt` 文件并生成 Document**

在 `app/documents.py` 写入：

```python
from pathlib import Path

from langchain_core.documents import Document


CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def load_documents(data_directory: Path) -> list[Document]:
    documents: list[Document] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for file_path in sorted(data_directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        for chunk_index, start in enumerate(range(0, len(text), step)):
            chunk = text[start : start + CHUNK_SIZE]
            if chunk:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": f"{file_path.name}#chunk-{chunk_index}"},
                    )
                )
    return documents
```

- [ ] **Step 4: 运行单测，确认通过**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py::test_load_documents_preserves_chunk_source_and_overlap -q
```

预期：`1 passed`。

- [ ] **Step 5: 补空资料测试并运行整个测试文件**

追加测试：

```python
def test_load_documents_returns_empty_list_for_empty_directory(tmp_path):
    assert load_documents(tmp_path) == []
```

运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py -q
```

预期：`2 passed`。

- [ ] **Step 6: 提交 Document 加载器**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git add app\documents.py tests\test_retriever.py
git commit -m "feat: add langchain document loader"
```

### Task 3: 先用 TDD 定义 DashScope 的 LangChain Embeddings 适配器

**Files:**
- Modify: `tests/test_retriever.py`
- Create: `app/embeddings.py`

**Interfaces:**
- Consumes: OpenAI-compatible client 的 `embeddings.create(model=..., input=list[str])` 响应，及 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`。
- Produces: `DashScopeEmbeddings(api_key: str | None = None, client: object | None = None)`，实现 `embed_documents(texts: list[str]) -> list[list[float]]` 与 `embed_query(text: str) -> list[float]`。

- [ ] **Step 1: 写失败测试，描述文档和问题向量方法**

在 `tests/test_retriever.py` 追加：

```python
from app.embeddings import DashScopeEmbeddings


class FakeEmbeddingsClient:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, input))
        rows = [type("Row", (), {"embedding": [float(index), 1.0]})() for index, _ in enumerate(input)]
        return type("Response", (), {"data": rows})()


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsClient()


def test_dashscope_embeddings_implements_langchain_methods():
    client = FakeClient()
    embeddings = DashScopeEmbeddings(client=client, api_key="test-key")

    assert embeddings.embed_documents(["第一段", "第二段"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert embeddings.embed_query("问题") == [0.0, 1.0]
    assert client.embeddings.calls == [
        ("text-embedding-v4", ["第一段", "第二段"]),
        ("text-embedding-v4", ["问题"]),
    ]
```

- [ ] **Step 2: 运行该测试，确认适配器尚不存在**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py::test_dashscope_embeddings_implements_langchain_methods -q
```

预期：FAIL，错误为 `ModuleNotFoundError: No module named 'app.embeddings'`。

- [ ] **Step 3: 写最小 Embeddings 适配器**

在 `app/embeddings.py` 写入：

```python
import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI


MODEL_NAME = "text-embedding-v4"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

load_dotenv()


class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str | None = None, client: object | None = None):
        resolved_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
        if not resolved_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法生成 Embedding。")
        self.client = client or OpenAI(
            api_key=resolved_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=MODEL_NAME, input=texts)
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
```

- [ ] **Step 4: 运行适配器测试，确认通过且未联网**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py::test_dashscope_embeddings_implements_langchain_methods -q
```

预期：`1 passed`；Fake Client 已拦截全部调用。

- [ ] **Step 5: 写并验证缺少 Key 的失败路径**

追加测试：

```python
import pytest


def test_dashscope_embeddings_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        DashScopeEmbeddings(api_key="")
```

运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py -q
```

预期：`4 passed`。

- [ ] **Step 6: 提交 Embeddings 适配器**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git add app\embeddings.py tests\test_retriever.py
git commit -m "feat: add dashscope langchain embeddings"
```

### Task 4: 先用 TDD 建立标准 LangChain Retriever

**Files:**
- Modify: `tests/test_retriever.py`
- Create: `app/retriever.py`

**Interfaces:**
- Consumes: `load_documents(data_directory)` 与任意 LangChain `Embeddings` 实现。
- Produces: `build_retriever(data_directory: Path, embeddings: Embeddings, top_k: int = 3) -> VectorStoreRetriever`；资料为空时抛出 `ValueError("没有可索引资料。")`。

- [ ] **Step 1: 写失败测试，定义 Fake Embeddings 与 Top-3 结果契约**

在 `tests/test_retriever.py` 追加：

```python
from langchain_core.embeddings import Embeddings

from app.retriever import build_retriever


class DeterministicEmbeddings(Embeddings):
    vectors = {
        "Python 文件安全": [1.0, 0.0],
        "Agent 权限确认": [0.0, 1.0],
        "RAG 文档切分": [0.8, 0.2],
        "如何确认副作用？": [0.0, 1.0],
    }

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        return self.vectors[text]


def test_build_retriever_returns_relevant_documents_with_source(tmp_path):
    (tmp_path / "python.txt").write_text("Python 文件安全", encoding="utf-8")
    (tmp_path / "agent.txt").write_text("Agent 权限确认", encoding="utf-8")
    (tmp_path / "rag.txt").write_text("RAG 文档切分", encoding="utf-8")

    retriever = build_retriever(tmp_path, DeterministicEmbeddings())
    results = retriever.invoke("如何确认副作用？")

    assert [document.metadata["source"] for document in results] == [
        "agent.txt#chunk-0",
        "rag.txt#chunk-0",
        "python.txt#chunk-0",
    ]
    assert [document.page_content for document in results] == [
        "Agent 权限确认",
        "RAG 文档切分",
        "Python 文件安全",
    ]
```

- [ ] **Step 2: 运行测试，确认构建函数尚不存在**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py::test_build_retriever_returns_relevant_documents_with_source -q
```

预期：FAIL，错误为 `ModuleNotFoundError: No module named 'app.retriever'`。

- [ ] **Step 3: 写最小 InMemoryVectorStore 实现**

在 `app/retriever.py` 写入：

```python
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.documents import load_documents


def build_retriever(
    data_directory: Path,
    embeddings: Embeddings,
    top_k: int = 3,
):
    documents = load_documents(data_directory)
    if not documents:
        raise ValueError("没有可索引资料。")

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
    return vector_store.as_retriever(search_kwargs={"k": top_k})
```

- [ ] **Step 4: 运行 Retriever 测试，确认 Top-3、内容与来源都通过**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py::test_build_retriever_returns_relevant_documents_with_source -q
```

预期：`1 passed`。

- [ ] **Step 5: 写空资料测试并运行完整测试文件**

追加测试：

```python
def test_build_retriever_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="没有可索引资料"):
        build_retriever(tmp_path, DeterministicEmbeddings())
```

运行：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py -q
```

预期：`6 passed`。

- [ ] **Step 6: 提交标准 Retriever**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
git add app\retriever.py tests\test_retriever.py
git commit -m "feat: add in-memory langchain retriever"
```

### Task 5: 添加命令行真实演示与学习 README

**Files:**
- Create: `main.py`
- Modify: `README.md`
- Modify: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `DashScopeEmbeddings`、`build_retriever`、`data/` 与位置参数 `question`。
- Produces: `python main.py "问题"`；打印每个 Document 的 `source` 和正文，并在空白问题时退出码为 2。

- [ ] **Step 1: 写失败测试，定义空白问题的命令行边界**

在 `tests/test_retriever.py` 追加：

```python
from main import validate_question


def test_validate_question_rejects_blank_input():
    with pytest.raises(ValueError, match="问题不能为空"):
        validate_question("  ")


def test_validate_question_strips_whitespace():
    assert validate_question("  如何确认副作用？  ") == "如何确认副作用？"
```

- [ ] **Step 2: 运行测试，确认命令行辅助函数尚不存在**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest tests\test_retriever.py::test_validate_question_rejects_blank_input -q
```

预期：FAIL，错误为 `ModuleNotFoundError: No module named 'main'`。

- [ ] **Step 3: 写最小命令行演示**

创建 `main.py`：

```python
import sys
from pathlib import Path

from app.embeddings import DashScopeEmbeddings
from app.retriever import build_retriever


DATA_DIRECTORY = Path(__file__).parent / "data"


def validate_question(question: str) -> str:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("问题不能为空。")
    return normalized_question


def main() -> int:
    try:
        question = validate_question(" ".join(sys.argv[1:]))
        retriever = build_retriever(DATA_DIRECTORY, DashScopeEmbeddings())
        for document in retriever.invoke(question):
            print(f"=== {document.metadata['source']} ===")
            print(document.page_content)
            print()
        return 0
    except ValueError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行完整离线测试**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest -q
```

预期：`8 passed`。

- [ ] **Step 5: 写 README 运行说明**

在 `README.md` 写明：

1. Week07 第一节目标与 Week06 的对照关系。
2. `Document`、`Embeddings`、`InMemoryVectorStore`、Retriever 分别承担什么职责。
3. Windows CMD 创建虚拟环境、安装依赖、运行测试、复制 `.env.example` 为 `.env` 的命令。
4. 真实演示命令：

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
copy .env.example .env
notepad .env
.venv\Scripts\python.exe main.py "如何确认副作用操作？"
```

5. `.env` 只填写真实 `DASHSCOPE_API_KEY`；它被 `.gitignore` 排除，不能提交或截图。
6. 内存向量库在程序退出后清空；本节没有 FastAPI、Docker、持久化向量库或记忆功能。

- [ ] **Step 6: 人工进行一次真实 DashScope 检索演示**

前提：`.env` 中已私下填入真实 `DASHSCOPE_API_KEY`。

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe main.py "如何确认副作用操作？"
```

预期：输出最多 3 个 `=== 文件名#chunk-N ===` 标题及对应资料正文。不得复制、提交或截图 API Key。

- [ ] **Step 7: 最终验证与提交**

```cmd
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week07-langchain-memory"
.venv\Scripts\python.exe -m pytest -q
git status --short
git add main.py README.md tests\test_retriever.py
git commit -m "docs: add retriever demo guide"
git status --short
```

预期：测试全绿；最后工作区干净，`.env` 仍未被 Git 跟踪。

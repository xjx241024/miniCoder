"""RAG 工具测试：schema、成功路径、错误码与主循环集成（全部离线）。"""

from core.config import EmbeddingConfig
from core.llm import ChatResponse
from core.message import FunctionCall, ToolCall
from rag.store import VectorStore
from runtime.loop import AgentLoop
from tools.builtin.index_docs_tool import IndexDocsTool
from tools.builtin.search_docs_tool import SearchDocsTool
from tools.registry import ToolRegistry
from tools.workspace import Workspace


class KeywordEmbedder:
    """确定性假向量：按关键词计数构造 2 维向量，模拟语义相关性。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(text.count("苹果")), float(text.count("香蕉"))] for text in texts]


def _make_tools(tmp_path):
    """构造共享同一索引库的两个工具（backend_factory 每次开新连接）。"""
    workspace = Workspace(tmp_path)
    embedder = KeywordEmbedder()
    db_path = tmp_path / "index.db"

    def factory():
        return embedder, VectorStore(db_path, model_id="test-model")

    index_tool = IndexDocsTool(workspace, backend_factory=factory)
    search_tool = SearchDocsTool(workspace, backend_factory=factory)
    return workspace, index_tool, search_tool


def test_tool_schemas():
    workspace = Workspace(".")
    index_tool = IndexDocsTool(workspace)
    search_tool = SearchDocsTool(workspace)
    assert index_tool.name == "index_docs"
    assert index_tool.schema()["function"]["parameters"]["required"] == ["path"]
    assert search_tool.name == "search_docs"
    assert search_tool.schema()["function"]["parameters"]["required"] == ["query"]


def test_index_docs_success_and_incremental(tmp_path):
    workspace, index_tool, _ = _make_tools(tmp_path)
    (tmp_path / "apple.md").write_text("苹果是一种水果。", encoding="utf-8")

    first = index_tool.invoke({"path": "apple.md"})
    second = index_tool.invoke({"path": "apple.md"})

    assert first.status == "success"
    assert first.data["indexed_files"] == 1
    assert "索引完成" in first.text
    assert second.status == "success"
    assert second.data["skipped_files"] == 1
    assert "跳过未变化 1" in second.text


def test_index_docs_rebuild_resets(tmp_path):
    workspace, index_tool, _ = _make_tools(tmp_path)
    (tmp_path / "apple.md").write_text("苹果内容", encoding="utf-8")

    index_tool.invoke({"path": "apple.md"})
    result = index_tool.invoke({"path": "apple.md", "rebuild": True})

    assert result.status == "success"
    assert result.data["indexed_files"] == 1
    assert result.data["skipped_files"] == 0


def test_search_docs_returns_hits_with_source(tmp_path):
    workspace, index_tool, search_tool = _make_tools(tmp_path)
    (tmp_path / "apple.md").write_text("苹果是一种水果，苹果很好吃。", encoding="utf-8")
    (tmp_path / "banana.md").write_text("香蕉是黄色的水果。", encoding="utf-8")

    index_tool.invoke({"path": "."})
    result = search_tool.invoke({"query": "苹果", "top_k": 2})

    assert result.status == "success"
    assert result.data["top_k"] == 2
    assert len(result.data["hits"]) == 2
    assert result.data["hits"][0]["path"] == "apple.md"
    assert "apple.md" in result.text
    assert "来源" in result.text


def test_search_docs_before_index_returns_index_empty(tmp_path):
    workspace, _, search_tool = _make_tools(tmp_path)
    result = search_tool.invoke({"query": "苹果"})
    assert result.status == "error"
    assert result.error.code == "INDEX_EMPTY"


def test_search_docs_clamps_top_k(tmp_path):
    workspace, index_tool, search_tool = _make_tools(tmp_path)
    (tmp_path / "apple.md").write_text("苹果内容", encoding="utf-8")
    index_tool.invoke({"path": "."})

    result = search_tool.invoke({"query": "苹果", "top_k": 999})
    assert result.status == "success"
    assert result.data["top_k"] == 20


def test_tools_report_embedding_not_configured(tmp_path, monkeypatch):
    """未配置 EMBEDDING_API_KEY 时，工具返回可诊断错误而非抛异常。"""
    import rag.backend as backend

    monkeypatch.setattr(
        backend, "load_embedding_config", lambda: EmbeddingConfig(api_key="")
    )
    workspace = Workspace(tmp_path)
    result = IndexDocsTool(workspace).invoke({"path": "."})
    assert result.status == "error"
    assert result.error.code == "EMBEDDING_NOT_CONFIGURED"
    assert ".env" in result.error.message


def test_registry_clean_arguments_for_rag_tools(tmp_path):
    """注册中心参数清洗：top_k 字符串应被归一为整数。"""
    workspace, index_tool, search_tool = _make_tools(tmp_path)
    (tmp_path / "apple.md").write_text("苹果内容", encoding="utf-8")
    registry = ToolRegistry(workspace=Workspace(tmp_path))
    registry.register(index_tool)
    registry.register(search_tool)

    registry.call("index_docs", {"path": "apple.md"})
    result = registry.call("search_docs", {"query": "苹果", "top_k": "3"})

    assert result.status == "success"
    assert result.data["top_k"] == 3


def test_agent_loop_integrates_rag_tools(tmp_path):
    """主循环集成：假模型先索引、再检索、最后基于检索结果回答。"""
    workspace, index_tool, search_tool = _make_tools(tmp_path)
    (tmp_path / "apple.md").write_text("苹果是一种水果，苹果很好吃。", encoding="utf-8")
    registry = ToolRegistry(workspace=Workspace(tmp_path))
    registry.register(index_tool)
    registry.register(search_tool)

    class FakeLLM:
        def __init__(self):
            self.responses = [
                ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        function=FunctionCall(name="index_docs", arguments='{"path": "."}'),
                    )],
                    finish_reason="tool_calls",
                ),
                ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_2",
                        function=FunctionCall(name="search_docs", arguments='{"query": "苹果"}'),
                    )],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="苹果是水果（来源：apple.md）"),
                ChatResponse(content="done"),
            ]

        def chat(self, messages, tools=None):
            return self.responses.pop(0)

    loop = AgentLoop(FakeLLM(), registry, max_steps=10)
    result = loop.run("苹果是什么？")

    assert result.success is True
    assert "apple.md" in result.answer
    tool_names = [step.name for step in result.trace if step.kind == "tool"]
    assert tool_names == ["index_docs", "search_docs"]

"""RAG 解析器测试：扩展名过滤、指纹计算、越界与异常路径。"""

from pathlib import Path

import pytest

from rag.parser import DEFAULT_MAX_CHARS, collect_files, parse_file
from tools.workspace import Workspace


def _make_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    root = tmp_path / "proj"
    root.mkdir()
    return Workspace(root), root


def test_parse_file_returns_document_with_fingerprint(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    target = root / "README.md"
    target.write_text("# 标题\n\n正文内容", encoding="utf-8")

    doc = parse_file(workspace, "README.md")

    assert doc.path == "README.md"
    assert doc.content == "# 标题\n\n正文内容"
    assert doc.size_bytes == target.stat().st_size
    assert doc.mtime_ms > 0
    assert len(doc.content_hash) == 64  # SHA-256 hex


def test_parse_file_rejects_unsupported_format(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    target = root / "manual.pdf"
    target.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(Exception) as exc_info:
        parse_file(workspace, "manual.pdf")
    assert exc_info.value.code == "UNSUPPORTED_FORMAT"


def test_parse_file_rejects_missing_and_outside(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    with pytest.raises(Exception) as exc_info:
        parse_file(workspace, "missing.md")
    assert exc_info.value.code == "NOT_FOUND"

    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(Exception) as exc_info:
        parse_file(workspace, str(outside))
    assert exc_info.value.code == "OUTSIDE_WORKSPACE"


def test_parse_file_rejects_too_large_file(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    target = root / "big.md"
    target.write_text("x" * (DEFAULT_MAX_CHARS + 1), encoding="utf-8")

    with pytest.raises(Exception) as exc_info:
        parse_file(workspace, "big.md")
    assert exc_info.value.code == "FILE_TOO_LARGE"


def test_collect_files_walks_directory_and_filters_noise(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "a.md").write_text("a", encoding="utf-8")
    (root / "docs" / "b.txt").write_text("b", encoding="utf-8")
    (root / "docs" / "c.pdf").write_bytes(b"pdf")
    (root / "app.py").write_text("print()", encoding="utf-8")
    noise = root / "node_modules" / "dep"
    noise.mkdir(parents=True)
    (noise / "dep.py").write_text("x", encoding="utf-8")

    files = collect_files(workspace, ".")

    rels = [f.relative_to(root).as_posix() for f in files]
    assert rels == ["app.py", "docs/a.md", "docs/b.txt"]


def test_collect_files_single_file_and_pattern_filter(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    (root / "a.md").write_text("a", encoding="utf-8")
    (root / "b.py").write_text("b", encoding="utf-8")

    assert collect_files(workspace, "a.md") == [root / "a.md"]
    assert collect_files(workspace, "a.md", pattern="*.py") == []
    assert collect_files(workspace, ".", pattern="*.py") == [root / "b.py"]

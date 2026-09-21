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
    target = root / "program.exe"
    target.write_bytes(b"MZ fake binary")

    with pytest.raises(Exception) as exc_info:
        parse_file(workspace, "program.exe")
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


def _make_pdf(path: Path, text: str) -> None:
    """构造最小可用 PDF（单页 Helvetica 文本），供 pypdf 提取验证。"""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    path.write_bytes(bytes(out))


def test_parse_pdf_extracts_text(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    _make_pdf(root / "manual.pdf", "Hello PDF RAG")

    doc = parse_file(workspace, "manual.pdf")

    assert "Hello PDF RAG" in doc.content
    assert doc.path == "manual.pdf"
    assert len(doc.content_hash) == 64


def test_parse_invalid_pdf_returns_parse_error(tmp_path):
    workspace, root = _make_workspace(tmp_path)
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 not a real pdf")

    with pytest.raises(Exception) as exc_info:
        parse_file(workspace, "broken.pdf")
    assert exc_info.value.code == "PARSE_ERROR"


def test_parse_docx_extracts_text(tmp_path):
    import docx

    workspace, root = _make_workspace(tmp_path)
    document = docx.Document()
    document.add_paragraph("DOCX RAG 测试内容")
    document.save(str(root / "report.docx"))

    doc = parse_file(workspace, "report.docx")

    assert "DOCX RAG 测试内容" in doc.content
    assert doc.path == "report.docx"


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
    (root / "docs" / "c.exe").write_bytes(b"MZ")
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

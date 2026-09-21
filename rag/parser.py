"""文档解析：按扩展名白名单读取文本 / PDF / DOCX，产出统一的 Document（M11）。

文本类文件直接 UTF-8 读取；PDF 用 pypdf 提取纯文本（按页拼接）；
DOCX 用 python-docx 提取段落文本。提取失败返回可诊断错误码。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from rag.errors import RAGError
from tools.workspace import Workspace, WorkspaceError

# 支持索引的文本扩展名：文档 + 常见代码 / 配置文件
SUPPORTED_EXTENSIONS = frozenset({
    ".md", ".markdown", ".mdx", ".txt", ".rst",
    ".pdf", ".docx",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".rb", ".php", ".sh", ".ps1", ".sql",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html", ".css",
})

# 递归索引时跳过的噪声目录（与 glob 工具的默认排除保持一致）
NOISE_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode", "dist", "build",
})

# 默认单文件最大字符数（约 200k 字符，防止超大文件拖慢索引）
DEFAULT_MAX_CHARS = 200_000


@dataclass
class Document:
    """一个待索引的文本文档；path 为工作区相对 posix 路径。"""

    path: str
    abs_path: Path
    content: str
    size_bytes: int
    mtime_ms: int
    content_hash: str


def parse_file(
    workspace: Workspace,
    requested: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Document:
    """读取一个受支持的文本文件，并计算指纹（大小 / mtime / SHA-256 内容哈希）。

    内容哈希用于增量索引：未变化的文件直接跳过，不重复消耗 Embedding 调用。
    """
    try:
        abs_path = workspace.resolve(requested)
    except WorkspaceError as exc:
        raise RAGError("OUTSIDE_WORKSPACE", exc.message) from exc
    if abs_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise RAGError(
            "UNSUPPORTED_FORMAT",
            f"不支持的文件类型: {abs_path.name}（首版支持文本/代码/配置文件，PDF 等后续支持）",
        )
    if not abs_path.is_file():
        raise RAGError("NOT_FOUND", f"文件不存在: {requested}")
    stat = abs_path.stat()
    # 先按字节粗检（UTF-8 最长 4 字节/字符），避免把超大文件整个读进内存
    if stat.st_size > max_chars * 4:
        raise RAGError("FILE_TOO_LARGE", f"文件过大: {abs_path.name}（上限约 {max_chars} 字符）")
    # 二进制文档格式（PDF / DOCX）：提取纯文本后统一走指纹与分块流程
    suffix = abs_path.suffix.lower()
    if suffix == ".pdf":
        content = _read_pdf(abs_path)
    elif suffix == ".docx":
        content = _read_docx(abs_path)
    else:
        content = _read_text(abs_path)
    if len(content) > max_chars:
        raise RAGError(
            "FILE_TOO_LARGE", f"文件过大: {abs_path.name}（{len(content)} 字符，上限 {max_chars}）"
        )
    return Document(
        path=workspace.relative(abs_path),
        abs_path=abs_path,
        content=content,
        size_bytes=stat.st_size,
        mtime_ms=int(stat.st_mtime * 1000),
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _read_text(abs_path: Path) -> str:
    """读取 UTF-8 文本文件；解码失败返回 DECODE_ERROR。"""
    try:
        return abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RAGError(
            "DECODE_ERROR", f"文件读取失败（可能不是 UTF-8 文本）: {abs_path.name}: {exc}"
        ) from exc


def _read_pdf(abs_path: Path) -> str:
    """用 pypdf 提取 PDF 纯文本；空文本/异常返回可诊断错误。"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(abs_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        content = "\n\n".join(pages).strip()
    except ImportError as exc:
        raise RAGError("DEPENDENCY_MISSING", "PDF 解析需要 pypdf：uv add pypdf") from exc
    except Exception as exc:
        raise RAGError("PARSE_ERROR", f"PDF 解析失败: {abs_path.name}: {exc}") from exc
    if not content:
        raise RAGError("PARSE_ERROR", f"PDF 未提取到文本（可能是扫描件）: {abs_path.name}")
    return content


def _read_docx(abs_path: Path) -> str:
    """用 python-docx 提取 DOCX 段落文本；异常返回可诊断错误。"""
    try:
        import docx

        document = docx.Document(str(abs_path))
        content = "\n\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    except ImportError as exc:
        raise RAGError(
            "DEPENDENCY_MISSING", "DOCX 解析需要 python-docx：uv add python-docx"
        ) from exc
    except Exception as exc:
        raise RAGError("PARSE_ERROR", f"DOCX 解析失败: {abs_path.name}: {exc}") from exc
    if not content:
        raise RAGError("PARSE_ERROR", f"DOCX 未提取到文本: {abs_path.name}")
    return content


def collect_files(
    workspace: Workspace,
    requested: str | Path,
    *,
    pattern: str = "*",
) -> list[Path]:
    """收集待索引文件：单文件直接返回；目录递归遍历并过滤扩展名 / 噪声目录。

    pattern 为 fnmatch 语法，作用于工作区相对路径；默认 "*" 匹配全部支持文件。
    返回排序后的绝对路径列表，保证索引顺序稳定（便于测试与 trace 复现）。
    """
    try:
        abs_path = workspace.resolve(requested)
    except WorkspaceError as exc:
        raise RAGError("OUTSIDE_WORKSPACE", exc.message) from exc
    if abs_path.is_file():
        if not fnmatch(workspace.relative(abs_path), pattern):
            return []
        return [abs_path]
    if not abs_path.is_dir():
        raise RAGError("NOT_FOUND", f"路径不存在: {requested}")
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(abs_path):
        # 原地剪枝噪声目录，避免深入 node_modules / .git 等
        dirnames[:] = [d for d in dirnames if d not in NOISE_DIRS]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            rel = workspace.relative(candidate)
            if fnmatch(rel, pattern):
                files.append(candidate)
    return sorted(files)

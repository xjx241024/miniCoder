"""运行期数据目录：默认 ~/.minicoder/<项目哈希>/，可用环境变量覆盖。

把 trace / transcript / artifact 从项目内 memory/ 迁到用户主目录，
避免污染被操作的项目仓库（仿 claude/codex 的数据存放方式）。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# 环境变量：覆盖默认数据根目录
DATA_DIR_ENV = "MINICODER_DATA_DIR"


def default_data_dir() -> Path:
    """返回数据根目录：环境变量优先，否则 ~/.minicoder。"""
    override = os.getenv(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".minicoder"


def project_key(workspace_root: str | Path) -> str:
    """按工作空间绝对路径生成稳定短哈希，作为项目数据子目录名。"""
    raw = str(Path(workspace_root).resolve())
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def project_dir(workspace_root: str | Path, data_dir: str | Path | None = None) -> Path:
    """返回某工作空间对应的项目数据目录。"""
    base = Path(data_dir) if data_dir else default_data_dir()
    return base / project_key(workspace_root)


def default_trace_path(session_id: str, *, workspace_root=None, data_dir=None) -> Path:
    """会话 trace 文件路径：<data_dir>/<project>/traces/<session_id>.jsonl。"""
    root = Path(workspace_root) if workspace_root else Path.cwd()
    return project_dir(root, data_dir) / "traces" / f"{session_id}.jsonl"


def default_transcript_path(session_id: str, *, workspace_root=None, data_dir=None) -> Path:
    """会话记录文件路径：<data_dir>/<project>/transcripts/<session_id>.jsonl。"""
    root = Path(workspace_root) if workspace_root else Path.cwd()
    return project_dir(root, data_dir) / "transcripts" / f"{session_id}.jsonl"


def default_artifact_dir(*, workspace_root=None, data_dir=None) -> Path:
    """工具大输出 artifact 目录：<data_dir>/<project>/artifacts/。"""
    root = Path(workspace_root) if workspace_root else Path.cwd()
    return project_dir(root, data_dir) / "artifacts"


def default_rag_dir(*, workspace_root=None, data_dir=None) -> Path:
    """RAG 索引库目录：<data_dir>/<project>/rag/（M11）。"""
    root = Path(workspace_root) if workspace_root else Path.cwd()
    return project_dir(root, data_dir) / "rag"

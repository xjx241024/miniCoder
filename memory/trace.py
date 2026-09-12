"""JSONL trace：把每一步运行记录增量写入文件，读取后可供回放与排查。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from memory.paths import default_trace_path  # noqa: F401  # 默认路径迁移到 ~/.minicoder


def new_session_id() -> str:
    """生成一次运行的会话 id：时间戳 + 随机后缀，保证可读且唯一。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


class TraceWriter:
    """把每一步追加写进 JSONL 文件；可用 with 语句自动关闭。"""

    def __init__(self, path: str | Path, session_id: str = "") -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(
        self,
        step: int,
        kind: str,
        *,
        name: str = "",
        detail: str = "",
        payload: dict | None = None,
    ) -> None:
        """写一条 trace 记录：时间、会话、步数、类型与附加数据。"""
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "step": step,
            "kind": kind,
            "name": name,
            "detail": detail,
            "payload": payload or {},
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """关闭底层文件。"""
        self._file.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def iter_records(path: str | Path) -> Iterator[dict]:
    """逐条读取 trace；损坏的行跳过，保证读取健壮。"""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_records(path: str | Path) -> list[dict]:
    """读取全部 trace 记录（按写入顺序）。"""
    return list(iter_records(path))


def format_record(record: dict) -> str:
    """把一条 trace 记录格式化成可读文本，供回放展示。"""
    kind = record.get("kind", "?")
    step = record.get("step", "?")
    name = record.get("name", "")
    if kind == "tool":
        status = (record.get("payload") or {}).get("result", {}).get("status", "?")
        return f"[step {step}] 工具 {name} → 状态 {status}"
    if kind == "answer":
        return f"[step {step}] 回答: {record.get('detail', '')}"
    return f"[step {step}] {kind} {name} {record.get('detail', '')}"


def replay(path: str | Path) -> str:
    """回放：把整个 trace 渲染成可读文本。"""
    return "\n".join(format_record(record) for record in iter_records(path))
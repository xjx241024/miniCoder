"""append-only 会话记录：把每次运行的消息逐条写入 JSONL，支持恢复继续会话。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from core.message import Message
from memory.paths import default_transcript_path  # noqa: F401  # 默认路径迁移到 ~/.minicoder


class TranscriptWriter:
    """把消息逐条追加进 JSONL 文件；可用 with 语句自动关闭。"""

    def __init__(self, path: str | Path, session_id: str = "") -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def append(self, message: Message) -> None:
        """追加一条消息（含时间戳与会话 id，便于按会话归档）。"""
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "message": message.to_api_dict(),
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """关闭底层文件。"""
        self._file.close()

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def iter_messages(path: str | Path) -> Iterator[Message]:
    """逐条读取会话记录；损坏的行跳过。"""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message")
            if message:
                yield Message(**message)


def load_messages(path: str | Path) -> list[Message]:
    """读回全部消息，供恢复会话使用。"""
    return list(iter_messages(path))
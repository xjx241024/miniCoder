"""memory 层测试：trace 写入/读取/回放，transcript 追加/恢复。"""

import json
from pathlib import Path

from core.message import assistant, user
from memory import trace as trace_mod
from memory.paths import default_data_dir
from memory.trace import TraceWriter, default_trace_path, load_records, new_session_id
from memory.transcript import TranscriptWriter, load_messages


def test_new_session_id_is_unique():
    """两次生成的会话 id 不应相同，且带时间戳前缀。"""
    first = new_session_id()
    second = new_session_id()
    assert first != second
    assert len(first) >= 8


def test_trace_writer_appends_and_loads(tmp_path):
    """trace 逐条追加为合法 JSON，读取后字段完整。"""
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path, session_id="s1") as writer:
        writer.write(1, "assistant", detail="思考中", payload={"x": 1})
        writer.write(2, "tool", name="glob", detail="status=success")
    records = load_records(path)
    assert len(records) == 2
    assert records[0]["session_id"] == "s1"
    assert records[0]["kind"] == "assistant"
    assert records[1]["name"] == "glob"
    # 每行都是可解析的 JSON
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)


def test_trace_writer_creates_parent_dir(tmp_path):
    """写入多级目录不存在时自动创建。"""
    path = tmp_path / "a" / "b" / "trace.jsonl"
    with TraceWriter(path, "s1") as writer:
        writer.write(1, "answer", detail="done")
    assert path.is_file()


def test_replay_renders_readable(tmp_path):
    """回放把 trace 渲染成可读文本。"""
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path, "s1") as writer:
        writer.write(1, "tool", name="glob", detail="status=success")
        writer.write(2, "answer", detail="完成")
    text = trace_mod.replay(path)
    assert "工具 glob" in text
    assert "回答: 完成" in text


def test_transcript_append_and_load(tmp_path):
    """transcript 追加消息后能原样读回。"""
    path = tmp_path / "session.jsonl"
    messages = [user("你好"), assistant("嗨")]
    with TranscriptWriter(path, "s1") as writer:
        for message in messages:
            writer.append(message)
    loaded = load_messages(path)
    assert [m.role for m in loaded] == ["user", "assistant"]
    assert loaded[0].content == "你好"


def test_default_trace_path_suffix():
    """默认 trace 路径以 session_id 命名 jsonl 文件。"""
    path = default_trace_path("abc123")
    assert path.name == "abc123.jsonl"


def test_default_data_dir_env_override(tmp_path, monkeypatch):
    """MINICODER_DATA_DIR 环境变量可覆盖默认数据目录。"""
    monkeypatch.setenv("MINICODER_DATA_DIR", str(tmp_path))
    assert default_data_dir() == tmp_path


def test_default_data_dir_defaults_to_home(monkeypatch):
    """未设置环境变量时回退到 ~/.minicoder。"""
    monkeypatch.delenv("MINICODER_DATA_DIR", raising=False)
    assert default_data_dir() == Path.home() / ".minicoder"

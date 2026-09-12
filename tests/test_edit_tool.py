"""Edit 工具与读后写保护的单元测试。"""

import os

from tools.builtin.edit_tool import EditTool
from tools.builtin.read_tool import ReadTool
from tools.registry import ToolRegistry
from tools.workspace import Workspace


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(EditTool())
    return registry


def _edit_args(file, old: str, new: str) -> dict:
    """构造 edit 工具参数。"""
    return {"path": str(file), "old_string": old, "new_string": new}


def test_edit_requires_read(tmp_path):
    """未 read 直接 edit 应被拒绝（FILE_NOT_READ）。"""
    file = tmp_path / "a.py"
    file.write_text("x = 1\n", encoding="utf-8")
    registry = _make_registry()
    result = registry.call("edit", _edit_args(file, "x = 1", "x = 2"))
    assert result.status == "error"
    assert result.error is not None and result.error.code == "FILE_NOT_READ"


def test_edit_success_after_read_and_cache_invalidated(tmp_path):
    """read 后 edit 成功；成功后缓存失效，再 edit 需重新 read。"""
    file = tmp_path / "a.py"
    file.write_text("x = 1\n", encoding="utf-8")
    registry = _make_registry()
    assert registry.call("read", {"path": str(file)}).status == "success"

    result = registry.call("edit", _edit_args(file, "x = 1", "x = 2"))
    assert result.status == "success"
    assert file.read_text(encoding="utf-8") == "x = 2\n"

    # 编辑成功后缓存被清空，未重新 read 的 edit 应被拒绝
    again = registry.call("edit", _edit_args(file, "x = 2", "x = 3"))
    assert again.status == "error"
    assert again.error is not None and again.error.code == "FILE_NOT_READ"


def test_edit_conflict_on_external_change(tmp_path):
    """read 后文件被外部修改，edit 应返回 CONFLICT。"""
    file = tmp_path / "a.py"
    file.write_text("x = 1\n", encoding="utf-8")
    registry = _make_registry()
    assert registry.call("read", {"path": str(file)}).status == "success"

    # 外部修改：改变文件大小，确保指纹变化
    file.write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = registry.call("edit", _edit_args(file, "x = 1", "x = 2"))
    assert result.status == "error"
    assert result.error is not None and result.error.code == "CONFLICT"


def test_edit_aba_conflict_detected_by_hash(tmp_path):
    """ABA 伪装：内容被改成等长的 B 且 mtime 被还原，mtime/size 指纹会漏判，
    内容哈希应仍返回 CONFLICT。"""
    file = tmp_path / "a.py"
    file.write_text("x = 1\n", encoding="utf-8")
    registry = _make_registry()
    assert registry.call("read", {"path": str(file)}).status == "success"

    # 记录原 mtime，然后改成等长内容 B 并把 mtime 还原 → mtime/size 与原指纹一致
    stat = file.stat()
    orig_mtime_ns = stat.st_mtime_ns
    file.write_text("x = 2\n", encoding="utf-8")
    os.utime(file, ns=(stat.st_atime_ns, orig_mtime_ns))

    result = registry.call("edit", _edit_args(file, "x = 2", "x = 3"))
    assert result.status == "error"
    assert result.error is not None and result.error.code == "CONFLICT"


def test_edit_old_string_not_found(tmp_path):
    """old_string 与文件内容不符时应返回 OLD_NOT_FOUND。"""
    file = tmp_path / "a.py"
    file.write_text("x = 1\n", encoding="utf-8")
    result = EditTool().invoke({"path": str(file), "old_string": "not exist", "new_string": "y"})
    assert result.status == "error"
    assert result.error is not None and result.error.code == "OLD_NOT_FOUND"


def test_edit_outside_workspace_rejected(tmp_path):
    """越界 edit 返回 OUTSIDE_WORKSPACE，而非误报 FILE_NOT_READ。"""
    registry = ToolRegistry(workspace=Workspace(tmp_path))
    registry.register(ReadTool(Workspace(tmp_path)))
    registry.register(EditTool(Workspace(tmp_path)))
    outside = tmp_path.parent / "outside.txt"
    result = registry.call("edit", _edit_args(str(outside), "a", "b"))
    assert result.status == "error"
    assert result.error is not None and result.error.code == "OUTSIDE_WORKSPACE"

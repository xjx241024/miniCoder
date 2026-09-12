"""工具注册中心：注册、schema 汇总、调用分发，内置文件读后写保护与参数清洗。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tools.base import BaseTool, ToolResult
from tools.workspace import Workspace, WorkspaceError

logger = logging.getLogger(__name__)

# 读后写保护依赖的固定工具名
READ_TOOL = "read"
EDIT_TOOL = "edit"
WRITE_TOOL = "write"

# 文件指纹：(mtime_ms, size_bytes, content_hash)
FileFingerprint = tuple[int, int, str]


class ToolRegistry:
    """按名字管理一组工具，统一向外提供 schema 与调用入口。

    内置读后写保护：read 成功后记录文件指纹（mtime/size/内容哈希），
    edit 前校验"是否读过"与"是否被外部修改"（乐观锁）。
    路径统一经工作空间解析，防止越界；调用前做参数清洗（类型归一/丢弃未知字段）。
    """

    def __init__(self, workspace: Workspace | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        # 文件指纹缓存：path -> FileFingerprint，来自最近一次成功 read
        self._read_cache: dict[str, FileFingerprint] = {}
        # 默认工作空间为当前目录，保证不传参数时行为与旧版一致
        self.workspace = workspace or Workspace(Path.cwd())

    def register(self, tool: BaseTool) -> None:
        """注册工具；同名工具不允许重复注册。"""
        if tool.name in self._tools:
            logger.warning("工具已存在，拒绝重复注册: %s", tool.name)
            raise ValueError(f"工具已注册: {tool.name}")
        self._tools[tool.name] = tool
        logger.info("工具已注册: %s", tool.name)

    def get(self, name: str) -> BaseTool:
        """按名字取工具，不存在时抛 KeyError。"""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"未注册的工具: {name}")
        return tool

    def names(self) -> list[str]:
        """返回所有工具名。"""
        return list(self._tools)

    def schemas(self) -> list[dict]:
        """返回发给模型的 tools 参数（OpenAI 函数调用格式）。"""
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: dict) -> ToolResult:
        """按名字调用工具；先做参数清洗，再对 read / edit 应用读后写保护。"""
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("调用未注册的工具: %s", name)
            return ToolResult.failure(code="UNKNOWN_TOOL", message=f"未注册的工具: {name}")

        arguments = self._clean_arguments(name, arguments)
        if name in (EDIT_TOOL, WRITE_TOOL):
            blocked = (
                self._check_edit_guard(arguments)
                if name == EDIT_TOOL
                else self._check_write_guard(arguments)
            )
            if blocked is not None:
                reason = blocked.error.code if blocked.error else ""
                logger.info("工具调用被拒 name=%s reason=%s", name, reason)
                return blocked

        start = time.monotonic()
        result = tool.invoke(arguments)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 维护读后写保护状态
        if name == READ_TOOL and result.status == "success":
            self._note_read(arguments, result.data)
        if name == WRITE_TOOL and result.status == "success":
            self._note_read(arguments, result.data)
        if name == EDIT_TOOL and result.status == "success":
            self._invalidate_read(result.data.get("path"))

        if result.status == "error":
            code = result.error.code if result.error else "UNKNOWN"
            logger.info("工具调用完成 name=%s status=error code=%s ms=%d", name, code, elapsed_ms)
        else:
            logger.info("工具调用完成 name=%s status=success ms=%d", name, elapsed_ms)
        return result

    # ---- 参数清洗 ----

    def _clean_arguments(self, name: str, arguments: dict) -> dict:
        """按工具 schema 归一参数类型，并丢弃未知字段（模型常传错类型或多余字段）。"""
        tool = self._tools.get(name)
        if tool is None or not isinstance(arguments, dict):
            return arguments
        props = (tool.parameters or {}).get("properties", {})
        cleaned: dict = {}
        for key, value in arguments.items():
            if key not in props:
                continue  # 未知字段直接丢弃
            cleaned[key] = _coerce_value(value, props[key].get("type", "string"))
        return cleaned

    # ---- 读后写保护 ----

    def _note_read(self, arguments: dict, data: dict) -> None:
        """记录最近一次成功 read 的文件指纹；路径键用工具解析后的绝对路径。"""
        required = ("mtime_ms", "size_bytes", "content_hash")
        path = data.get("path") or arguments.get("path")
        if not path or any(key not in data for key in required):
            return
        self._read_cache[str(path)] = (
            int(data["mtime_ms"]),
            int(data["size_bytes"]),
            str(data["content_hash"]),
        )

    def _invalidate_read(self, path: str | None) -> None:
        """文件被修改后清除指纹，强制下次重新 read。"""
        if path:
            self._read_cache.pop(str(path), None)

    def _check_edit_guard(self, arguments: dict) -> ToolResult | None:
        """edit 前置校验：必须先 read，并自动注入乐观锁参数。

        这是乐观锁的"检查"阶段，并非互斥锁：只做冲突检测，
        真正写入由 EditTool 用 os.replace 原子完成。
        """
        try:
            key = str(self.workspace.resolve(str(arguments.get("path", ""))))
        except WorkspaceError:
            # 越界路径交给工具层统一拒绝；这里不查指纹，
            # 否则会误报"文件未读取"（错误信息与真实原因不匹配）
            return None
        fingerprint = self._read_cache.get(key)
        if fingerprint is None:
            return ToolResult.failure(
                code="FILE_NOT_READ",
                message=(
                    f"文件未读取: {arguments.get('path', '')}，请先 read 再 edit"
                ),
            )
        # 注入 read 时记录的指纹，模型无需自己传
        arguments["mtime_ms"] = fingerprint[0]
        arguments["size_bytes"] = fingerprint[1]
        arguments["content_hash"] = fingerprint[2]
        return None

    def _check_write_guard(self, arguments: dict) -> ToolResult | None:
        """write 前置校验：覆盖已存在文件必须先 read（防止盲写覆盖）。

        新建文件无需 read；已存在文件会注入 read 时记录的指纹，
        由 WriteTool 写入前做乐观锁冲突检测。
        """
        try:
            key = str(self.workspace.resolve(str(arguments.get("path", ""))))
        except WorkspaceError:
            return None  # 越界路径交给工具层统一拒绝
        if not Path(key).is_file():
            return None  # 新建文件：无需 read
        fingerprint = self._read_cache.get(key)
        if fingerprint is None:
            return ToolResult.failure(
                code="FILE_NOT_READ",
                message=(
                    f"文件已存在且未读取: {arguments.get('path', '')}，"
                    "write 会覆盖整个文件，请先 read 再 write"
                ),
            )
        arguments["mtime_ms"] = fingerprint[0]
        arguments["size_bytes"] = fingerprint[1]
        arguments["content_hash"] = fingerprint[2]
        return None


def _coerce_value(value: object, type_name: str) -> object:
    """按 JSON Schema 类型归一单个参数；转换失败时保留原值。"""
    if type_name == "string":
        return value if isinstance(value, str) else str(value)
    if type_name == "integer":
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if type_name == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if type_name == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes")
        return bool(value)
    if type_name == "array" and not isinstance(value, list):
        return [value]
    return value

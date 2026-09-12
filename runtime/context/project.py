"""项目规则发现（L2 静态层）：从工作目录向上读取 AGENTS.md / .minicoder/context.md。"""

from __future__ import annotations

from pathlib import Path

# 自动发现的规则文件（项目约定），作为"项目规则"优先
AGENT_FILES = ("AGENTS.md", "CLAUDE.md")

# 用户自建的自定义上下文文件，与 AGENTS.md 同时存在时合并（AGENTS.md 在前）
CUSTOM_FILE = ".minicoder/context.md"


class ProjectRulesLoader:
    """向上逐级查找项目规则文件，并把找到的规则合并成一段 L2 文本。"""

    def __init__(self, cwd: str | Path) -> None:
        self.cwd = Path(cwd)

    def load(self) -> str | None:
        """返回合并后的项目规则文本；未找到任何规则文件时返回 None。"""
        blocks: list[str] = []
        agent = self._find_any_up(AGENT_FILES)
        if agent is not None:
            blocks.append(f"## 项目规则（来自 {agent.name}）\n\n{_read_text(agent)}")
        custom = self._find_any_up((CUSTOM_FILE,))
        if custom is not None:
            blocks.append(f"## 项目自定义上下文（来自 {custom}）\n\n{_read_text(custom)}")
        if not blocks:
            return None
        return "\n\n".join(blocks)

    def _find_any_up(self, rel_paths: tuple[str, ...]) -> Path | None:
        """从 cwd 向上逐级查找第一个存在的规则文件（相对路径元组）。

        到达 git 仓库根（存在 .git）即停止，避免把上级无关仓库的规则带进来。
        """
        current = self.cwd
        while True:
            for rel in rel_paths:
                candidate = current / rel
                if candidate.is_file():
                    return candidate
            if (current / ".git").exists():
                return None
            if current.parent == current:
                return None
            current = current.parent


def _read_text(path: Path) -> str:
    """读取规则文件文本，异常时返回空串（不阻塞主流程）。"""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
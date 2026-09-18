"""文本分块器：递归字符分割 + 重叠（overlap），对中文文本友好（M11）。

纯函数实现、不依赖网络与分词器：先按分隔符优先级（段落 → 换行 → 句号 → 空格 → 硬切）
切成原子片段，再贪心合并到目标块大小；相邻块通过尾部 overlap 保持上下文连续。
"""

from __future__ import annotations

from dataclasses import dataclass

# 分隔符优先级：越靠前越优先（尽量在自然边界切分）
_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", ". ", "; ", " ")


@dataclass
class Chunk:
    """一个文本分块；char_start/char_end 指向原文中"新增内容"的区间。

    注意：content 的开头可能包含与上一块重叠的 overlap 文本，
    因此 len(content) 可以大于 char_end - char_start。
    """

    index: int
    content: str
    char_start: int
    char_end: int


class Chunker:
    """递归字符分割器：chunk_size 为目标块大小（字符），overlap 为相邻块重叠字符数。"""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正数: {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(f"overlap 必须满足 0 <= overlap < chunk_size: {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        """把长文本切成带重叠的块；空白文本返回空列表。"""
        text = text or ""
        if not text.strip():
            return []
        pieces = self._recursive_split(text, self.chunk_size)
        return self._merge_with_overlap(pieces)

    def _recursive_split(self, text: str, size: int) -> list[str]:
        """递归按分隔符切分：返回的片段都不超过 size 字符。"""
        if len(text) <= size:
            return [text] if text else []
        for separator in _SEPARATORS:
            if separator not in text:
                continue
            parts = text.split(separator)
            pieces: list[str] = []
            # 分隔符保留在前一段末尾，保证片段按顺序拼接后等于原文（便于追踪偏移）
            for i, part in enumerate(parts):
                piece = part + separator if i < len(parts) - 1 else part
                if not piece:
                    continue
                pieces.append(piece)
            # 分隔符只出现在文本末尾时，切分结果仍是原文本；
            # 此时换下一个分隔符，避免对同一文本无限递归
            if len(pieces) == 1 and len(pieces[0]) == len(text):
                continue
            # 递归切分超长片段（多个片段时每个必然小于原文，递归必然收敛）
            result: list[str] = []
            for piece in pieces:
                if len(piece) <= size:
                    result.append(piece)
                else:
                    result.extend(self._recursive_split(piece, size))
            return result
        # 所有分隔符都未命中（如超长无空格串）：按长度硬切
        return [text[i : i + size] for i in range(0, len(text), size)]

    def _merge_with_overlap(self, pieces: list[str]) -> list[Chunk]:
        """把原子片段贪心合并成块，并在相邻块之间注入尾部 overlap。"""
        chunks: list[Chunk] = []
        current = ""            # 当前块内容（含可能的 overlap 前缀）
        current_start = 0       # 当前块"新增内容"在原文中的起点
        consumed = 0            # 已消费的原文长度
        for piece in pieces:
            fits = not current or len(current) + len(piece) <= self.chunk_size
            if not fits:
                chunks.append(Chunk(len(chunks), current, current_start, consumed))
                # overlap 取上一块尾部，且保证 tail + piece 不超过块大小
                max_tail = min(self.overlap, self.chunk_size - len(piece))
                tail = current[-max_tail:] if max_tail > 0 else ""
                current = tail + piece
                current_start = consumed
                consumed += len(piece)
            else:
                if not current:
                    current_start = consumed
                current += piece
                consumed += len(piece)
        if current:
            chunks.append(Chunk(len(chunks), current, current_start, consumed))
        return chunks

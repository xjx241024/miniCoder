"""RAG 分块器测试：块大小、overlap、偏移覆盖与中文边界。"""

import pytest

from rag.chunking import Chunker


def test_short_text_returns_single_chunk():
    chunks = Chunker(chunk_size=100, overlap=10).split("你好，世界")
    assert len(chunks) == 1
    assert chunks[0].content == "你好，世界"
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len("你好，世界")


def test_empty_text_returns_empty_list():
    assert Chunker().split("") == []
    assert Chunker().split("   \n  ") == []


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0)
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, overlap=-1)


def test_all_chunks_respect_size_limit():
    text = "\n\n".join(f"第{i}段内容。" * 20 for i in range(30))
    chunks = Chunker(chunk_size=120, overlap=20).split(text)
    assert len(chunks) > 1
    assert all(len(c.content) <= 120 for c in chunks)


def test_consecutive_chunks_share_overlap():
    text = "\n".join(f"line-{i} " + "x" * 20 for i in range(40))
    chunks = Chunker(chunk_size=100, overlap=15).split(text)
    assert len(chunks) > 1
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        assert curr.content.startswith(prev.content[-15:])


def test_offsets_cover_whole_text():
    text = "。".join(f"句子{i}" for i in range(100))
    chunks = Chunker(chunk_size=30, overlap=5).split(text)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    # 相邻块的新增区间应无缝衔接（不重不漏）
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        assert curr.char_start == prev.char_end


def test_hard_split_without_any_separator():
    text = "字" * 250  # 无换行/句号/空格，触发硬切
    chunks = Chunker(chunk_size=100, overlap=10).split(text)
    assert len(chunks) == 3
    assert all(len(c.content) <= 100 for c in chunks)
    # 用偏移校验"新增内容"区间：每块 content = overlap 前缀 + 原文切片
    assert (chunks[0].char_start, chunks[0].char_end) == (0, 100)
    assert (chunks[1].char_start, chunks[1].char_end) == (100, 200)
    assert (chunks[2].char_start, chunks[2].char_end) == (200, 250)
    for chunk in chunks:
        assert chunk.content.endswith(text[chunk.char_start : chunk.char_end])
    # 片段小于块大小时 overlap 才放得下；整块填满时自动省略（保证不超块上限）
    assert chunks[2].content.startswith(chunks[1].content[-10:])

"""检索质量评测测试：Hit@K 与 MRR 计算。"""

import pytest

from rag.evaluation import EvalCase, evaluate_retrieval
from rag.store import SearchHit


def _hit(path: str) -> SearchHit:
    return SearchHit(path, 0, f"内容-{path}", 0, 5, 0.9, ["hybrid"])


def test_evaluate_computes_hit_rate_and_mrr():
    cases = [
        EvalCase("q1", ["a.md"]),
        EvalCase("q2", ["b.md"]),
        EvalCase("q3", ["missing.md"]),
    ]

    def search_fn(query: str, top_k: int) -> list[SearchHit]:
        # q1: a 排第 1；q2: b 排第 2；q3: 永远不命中
        if query == "q1":
            return [_hit("a.md"), _hit("x.md")][:top_k]
        if query == "q2":
            return [_hit("x.md"), _hit("b.md")][:top_k]
        return [_hit("x.md")][:top_k]

    report = evaluate_retrieval(search_fn, cases, top_k=2)

    assert report.total == 3
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert report.cases[0].rank == 1
    assert report.cases[1].rank == 2
    assert report.cases[2].hit is False
    assert report.cases[2].reciprocal_rank == 0.0


def test_evaluate_empty_cases_returns_zero_metrics():
    report = evaluate_retrieval(lambda query, top_k: [], [], top_k=5)
    assert report.total == 0
    assert report.hit_rate == 0.0
    assert report.mrr == 0.0


def test_evaluate_top_k_truncates_retrieved_paths():
    def search_fn(query: str, top_k: int) -> list[SearchHit]:
        return [_hit("a.md"), _hit("b.md"), _hit("c.md")]

    report = evaluate_retrieval(search_fn, [EvalCase("q", ["a.md"])], top_k=2)
    assert report.cases[0].retrieved_paths == ["a.md", "b.md"]
    assert report.cases[0].hit is True

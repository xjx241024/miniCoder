"""检索质量评测：Hit@K（Recall@K）与 MRR（M11 增强）。

评测口径：retrieved top-K 的来源路径中是否包含标注的相关文档（Hit@K），
以及第一个相关文档的排名倒数（MRR）。相关粒度为文档路径级，
足够反映"该找的文件有没有被召回、排得多靠前"。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from rag.store import SearchHit

# 检索函数签名：query + top_k -> 命中列表（与 DocRetriever.search 对齐）
SearchFn = Callable[[str, int], list[SearchHit]]


@dataclass
class EvalCase:
    """一条评测样例：查询 + 相关文档路径集合。"""

    query: str
    relevant_paths: list[str]


@dataclass
class CaseResult:
    """单条样例的评测结果。"""

    query: str
    hit: bool
    rank: int | None
    reciprocal_rank: float
    retrieved_paths: list[str]


@dataclass
class RetrievalEvalReport:
    """整组评测的汇总报告。"""

    top_k: int
    total: int
    hit_rate: float  # Hit@K = 至少命中一个相关文档的比例
    mrr: float       # Mean Reciprocal Rank
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典。"""
        return asdict(self)


def evaluate_retrieval(
    search_fn: SearchFn,
    cases: list[EvalCase],
    *,
    top_k: int = 5,
) -> RetrievalEvalReport:
    """跑一组评测样例并汇总 Hit@K 与 MRR。"""
    results: list[CaseResult] = []
    for case in cases:
        # 强制截断 top_k：不信任检索函数一定遵守上限，保证指标口径一致
        hits = search_fn(case.query, top_k)[:top_k]
        retrieved = [hit.path for hit in hits]
        rank = next(
            (index + 1 for index, path in enumerate(retrieved) if path in case.relevant_paths),
            None,
        )
        hit = rank is not None
        results.append(
            CaseResult(
                query=case.query,
                hit=hit,
                rank=rank,
                reciprocal_rank=1.0 / rank if hit else 0.0,
                retrieved_paths=retrieved,
            )
        )
    total = len(results)
    hit_rate = sum(1 for result in results if result.hit) / total if total else 0.0
    mrr = sum(result.reciprocal_rank for result in results) / total if total else 0.0
    return RetrievalEvalReport(
        top_k=top_k,
        total=total,
        hit_rate=hit_rate,
        mrr=mrr,
        cases=results,
    )

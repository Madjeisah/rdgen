"""Context strategies: how a (possibly perturbed) document becomes model input.

Perturbations always act on the full document first; the context strategy is
applied afterwards, so trimming never silently re-deletes evidence that the
estimator thinks is present (that interaction is reported, not hidden).
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Item


def approx_tokens(s: str) -> int:
    return max(1, int(len(s.split()) * 1.35))


@dataclass
class ContextConfig:
    strategy: str = "trim"      # trim | retrieve | full
    max_tokens: int = 2048
    top_k: int = 10             # for retrieve


def trim(units: list[str], max_tokens: int) -> list[int]:
    keep, used = [], 0
    for i, u in enumerate(units):
        t = approx_tokens(u)
        if used + t > max_tokens:
            break
        keep.append(i)
        used += t
    return keep


def retrieve(units: list[str], query: str, top_k: int) -> list[int]:
    from rank_bm25 import BM25Okapi
    if not units:
        return []
    tok = [u.lower().split() for u in units]
    scores = BM25Okapi(tok).get_scores(query.lower().split())
    best = sorted(range(len(units)), key=lambda i: -scores[i])[:top_k]
    return sorted(best)                     # keep document order


def select_units(item: Item, cfg: ContextConfig) -> list[int]:
    if cfg.strategy == "full":
        return list(range(len(item.units)))
    if cfg.strategy == "trim":
        return trim(item.units, cfg.max_tokens)
    if cfg.strategy == "retrieve":
        return retrieve(item.units, item.question, cfg.top_k)
    raise ValueError(f"unknown context strategy {cfg.strategy!r}")


def render(item: Item, cfg: ContextConfig, numbered: bool = True) -> str:
    idx = select_units(item, cfg)
    if numbered:
        return "\n".join(f"[{i}] {item.units[i]}" for i in idx)
    return " ".join(item.units[i] for i in idx)


def evidence_visible(item: Item, cfg: ContextConfig) -> float:
    """Share of gold evidence units that survive the context strategy."""
    if not item.evidence:
        return float("nan")
    keep = set(select_units(item, cfg))
    return sum(i in keep for i in item.evidence) / len(item.evidence)

"""Record-dependence estimators (methodology 10.3-10.4).

``probe_model`` runs one answerer on the original, evidence-deleted,
control-deleted, sufficiency and substitution versions of each item, in a
single batched call, and returns one row per item. ``summarize_probe``
turns those rows into the model-level quantities:

  RD     = P[answer changes | evidence deleted, correct]
         - P[answer changes | control deleted,  correct]       (D+ items only)
  RD_raw = the first term alone (ablation: no control)
  SUF    = P[answer kept | evidence + filler only, correct]
  TR     = P[answer follows the substituted value | correct]
  UC     = P[still correct after deletion | correct]           (with chance rate)
  AB     = P[abstains after deletion | correct]

Confidence intervals use a cluster bootstrap over ``group_id`` (patients or
documents), because items from one record are correlated.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .models.base import Answerer
from .normalize import is_abstention, is_correct, normalize_text, same_answer, token_f1
from .perturb import (build_filler_pool, control_indices, delete, substitute,
                      sufficiency_item, support_closure)
from .schema import CATEGORICAL_TYPE, Item
from .verify import Verifier


@dataclass
class ProbeConfig:
    closure_mode: str = "full"
    tau: float = 0.5
    use_control: bool = True
    do_sufficiency: bool = True
    do_substitution: bool = True
    include_polarities: tuple[str, ...] = ("pos", "explicit_neg", "span")
    # Deletion is only a valid dependence test where removing the evidence
    # changes what a faithful reader should answer. Deleting an explicit
    # negation leaves a closed-world reader answering "no", so explicit_neg
    # items are measured by substitution (TR) instead. Set to include
    # "explicit_neg" only as the ablation that demonstrates this.
    deletion_polarities: tuple[str, ...] = ("pos", "span")
    # How the control is removed from the evidence-deletion change rate
    # (c = control change rate, p = evidence-deletion change rate):
    #   union      (p - c) / (1 - c)   evidence and disruption act as independent
    #                                  causes and disruption never undoes a change
    #   difference  p - c              first-order approximation, small effects only
    #   xor        (p - c) / (1 - 2c)  binary flips that can cancel each other
    control_correction: str = "union"
    seed: int = 0


def _alt_key(it: Item) -> str:
    # question template where one exists; otherwise the source dataset (e.g. the
    # MRQA sub-dataset), so substitutes come from the same kind of document
    return it.meta.get("template") or f"{it.answer_type}:{it.meta.get('source', '')}"


def _alt_answers(items: Sequence[Item]) -> dict[str, list[str]]:
    by = defaultdict(set)
    for it in items:
        by[_alt_key(it)].add(it.answer)
    return {k: sorted(v) for k, v in by.items()}


def probe_model(model: Answerer, items: Sequence[Item], cfg: ProbeConfig = ProbeConfig(),
                verifier: Verifier | None = None, filler_items: Sequence[Item] | None = None
                ) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    filler = build_filler_pool(filler_items or items, seed=cfg.seed)
    alts = _alt_answers(filler_items or items)

    batch: list[Item] = []
    plan: list[dict] = []
    for it in items:
        row = {"item_id": it.item_id, "group_id": it.group_id, "polarity": it.polarity,
               "answer_type": it.answer_type, "gold": it.answer,
               "template": it.meta.get("template", ""), "i_orig": len(batch)}
        batch.append(it)
        if it.polarity in cfg.include_polarities:
            clo = support_closure(it, cfg.closure_mode, verifier, cfg.tau)
            row["closure_size"] = len(clo)
            if clo and it.polarity in cfg.deletion_polarities:
                row["i_del"] = len(batch)
                batch.append(delete(it, clo, "evidence"))
                if cfg.use_control:
                    ctl = control_indices(it, clo, rng)
                    row["ctrl_shortfall"] = ctl.shortfall
                    if ctl.indices:
                        row["i_ctrl"] = len(batch)
                        batch.append(delete(it, ctl.indices, "control"))
                if cfg.do_sufficiency and filler:
                    row["i_suf"] = len(batch)
                    batch.append(sufficiency_item(it, clo, filler, rng))
            if clo:
                if cfg.do_substitution:
                    sub = substitute(it, clo, alts.get(_alt_key(it), []), rng)
                    if sub is not None:
                        row["i_sub"] = len(batch)
                        row["sub_expected"] = sub.expected
                        batch.append(sub.item)
        plan.append(row)

    preds = model.predict(batch)
    rows = []
    for r in plan:
        it_type = r["answer_type"]
        p0 = preds[r["i_orig"]]
        out = dict(r)
        out["pred"] = p0.answer
        out["confidence"] = p0.confidence
        out["correct"] = is_correct(p0.answer, r["gold"], it_type)
        if "i_del" in r:
            pd_ = preds[r["i_del"]].answer
            out["pred_del"] = pd_
            out["changed_del"] = not same_answer(pd_, p0.answer, it_type)
            out["correct_del"] = is_correct(pd_, r["gold"], it_type)
            out["abstain_del"] = is_abstention(pd_)
        if "i_ctrl" in r:
            out["changed_ctrl"] = not same_answer(preds[r["i_ctrl"]].answer, p0.answer, it_type)
        if "i_suf" in r:
            out["kept_suf"] = same_answer(preds[r["i_suf"]].answer, p0.answer, it_type)
        if "i_sub" in r:
            ps = preds[r["i_sub"]].answer
            out["pred_sub"] = ps
            out["tracked_sub"] = (same_answer(ps, r["sub_expected"], it_type) if it_type == CATEGORICAL_TYPE
                                  else normalize_text(ps) == normalize_text(r["sub_expected"]))
            # why a substitution was not followed: kept the original answer (a
            # prior or memorised answer, i.e. guessing), abstained, or other
            out["sub_outcome"] = ("followed" if out["tracked_sub"]
                                  else "kept_original" if same_answer(ps, p0.answer, it_type)
                                  else "abstained" if is_abstention(ps)
                                  else "followed_partly" if (it_type != CATEGORICAL_TYPE and
                                                             token_f1(ps, r["sub_expected"]) >= 0.5)
                                  else "other")
        for k in ("i_orig", "i_del", "i_ctrl", "i_suf", "i_sub"):
            out.pop(k, None)
        rows.append(out)
    return pd.DataFrame(rows)


def chance_rates(train_items: Sequence[Item]) -> dict[str, float]:
    """Majority-answer rate per template on the training split (UC reference)."""
    by: dict[str, Counter] = defaultdict(Counter)
    for it in train_items:
        by[it.meta.get("template", it.answer_type)][normalize_text(it.answer)] += 1
    return {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in by.items()}


def correct_for_control(p: float, c: float, mode: str = "union") -> float:
    if mode == "difference":
        return p - c
    if mode == "union":
        return (p - c) / (1 - c) if c < 1 else float("nan")
    if mode == "xor":
        return (p - c) / (1 - 2 * c) if abs(1 - 2 * c) > 1e-9 else float("nan")
    raise ValueError(f"unknown control_correction {mode!r}")


def _stats(df: pd.DataFrame, chance: dict[str, float] | None,
           correction: str = "union") -> dict[str, float]:
    out: dict[str, float] = {"n_items": len(df), "accuracy": float(df["correct"].mean()) if len(df) else np.nan}
    if "changed_del" in df:
        d = df[df["correct"].astype(bool) & df["changed_del"].notna()]
    else:
        d = df.iloc[0:0]
    out["n_rd"] = len(d)
    if len(d):
        raw = d["changed_del"].astype(float).mean()
        ctrl = d["changed_ctrl"].astype(float).mean() if "changed_ctrl" in d and d["changed_ctrl"].notna().any() else 0.0
        out["RD_raw"] = raw
        out["RD_ctrl_change"] = ctrl
        out["RD"] = correct_for_control(raw, ctrl, correction)
        out["RD_diff"] = raw - ctrl
        out["RD_xor"] = correct_for_control(raw, ctrl, "xor")
        out["UC"] = d["correct_del"].astype(float).mean()
        out["AB"] = d["abstain_del"].astype(float).mean()
        if chance is not None:
            out["UC_chance"] = float(np.mean([chance.get(t, np.nan) for t in d["template"]]))
        if "kept_suf" in d and d["kept_suf"].notna().any():
            out["SUF"] = d["kept_suf"].dropna().astype(float).mean()
    if "tracked_sub" in df:
        t = df[df["correct"].astype(bool) & df["tracked_sub"].notna()]
        out["n_tr"] = len(t)
        if len(t):
            out["TR"] = t["tracked_sub"].astype(float).mean()
            if "sub_outcome" in t:
                for k in ("kept_original", "abstained", "followed_partly", "other"):
                    out[f"TR_{k}"] = float((t["sub_outcome"] == k).mean())
    return out


def summarize_probe(df: pd.DataFrame, chance: dict[str, float] | None = None,
                    n_boot: int = 1000, seed: int = 0, by: str | None = None,
                    correction: str = "union") -> pd.DataFrame:
    """Point estimates + 95% cluster-bootstrap CIs. ``by`` stratifies
    (e.g. 'polarity' or 'answer_type')."""
    groups = [("all", df)] if by is None else list(df.groupby(by))
    rng = np.random.default_rng(seed)
    records = []
    for name, sub in groups:
        point = _stats(sub, chance, correction)
        clusters = sub["group_id"].unique()
        idx = {g: np.flatnonzero(sub["group_id"].to_numpy() == g) for g in clusters}
        boots: dict[str, list[float]] = defaultdict(list)
        if n_boot and len(clusters) > 1:
            for _ in range(n_boot):
                pick = rng.choice(clusters, size=len(clusters), replace=True)
                rows = np.concatenate([idx[g] for g in pick])
                s = _stats(sub.iloc[rows], chance, correction)
                for k, v in s.items():
                    if isinstance(v, float) and not np.isnan(v):
                        boots[k].append(v)
        for k, v in point.items():
            rec = {"stratum": name, "metric": k, "estimate": v}
            if k in boots and len(boots[k]) > 10:
                rec["lo"], rec["hi"] = np.percentile(boots[k], [2.5, 97.5])
            records.append(rec)
    return pd.DataFrame(records)


def point(df_summary: pd.DataFrame, metric: str, stratum: str = "all") -> float:
    s = df_summary[(df_summary.metric == metric) & (df_summary.stratum == stratum)]
    return float(s.estimate.iloc[0]) if len(s) else float("nan")

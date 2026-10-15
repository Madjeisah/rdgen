"""Evaluate one model on one domain (protocol step 2).

Outputs, under ``<out>/<domain>/<model>/``:
  preds.csv     one row per item and split: prediction, confidence, correctness
  probe.csv     record-dependence probe rows on the in-domain test split
  summary.json  model-level quantities used by the analysis

The shifted split is only scored when the pre-registration lock is valid.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..baselines.atc import atc_predict, doc_predict
from ..baselines.contextcite import contextcite_dependence
from ..estimators import ProbeConfig, chance_rates, point, probe_model, summarize_probe
from ..models.base import Answerer
from ..normalize import is_correct
from ..schema import TEST_ID, TEST_SHIFT, TRAIN, VALID, Item
from ..verify import Verifier
from .prereg import require_lock


@dataclass
class ModelMeta:
    name: str
    family: str
    size_b: float = float("nan")
    regime: str = "sft"
    context: str = "trim"
    seed: int = 0
    base_model: str = ""
    extra: dict = field(default_factory=dict)


def _predict_split(model: Answerer, items: Sequence[Item], split: str) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    preds = model.predict(items)
    return pd.DataFrame([{
        "item_id": it.item_id, "group_id": it.group_id, "split": split, "polarity": it.polarity,
        "answer_type": it.answer_type, "gold": it.answer, "pred": p.answer,
        "confidence": p.confidence, "correct": is_correct(p.answer, it.answer, it.answer_type),
    } for it, p in zip(items, preds)])


def evaluate_model(model: Answerer, meta: ModelMeta, items: Sequence[Item], out_dir: str | Path,
                   probe_cfg: ProbeConfig = ProbeConfig(), verifier: Verifier | None = None,
                   prereg_path: str | None = None, lock_path: str = "prereg.lock",
                   override_prereg: bool = False, contextcite: bool = True,
                   n_boot: int = 500, ablations: bool = False,
                   alt_verifier: Verifier | None = None, ablation_max_items: int | None = None
                   ) -> dict:
    from ..ablation import CachingAnswerer, default_arms, run_measurement_ablations
    model = CachingAnswerer(model)          # one cache for predictions, probes and ablation arms
    by = {s: [it for it in items if it.split == s] for s in (TRAIN, VALID, TEST_ID, TEST_SHIFT)}
    domain = items[0].domain if items else "unknown"
    out = Path(out_dir) / domain / meta.name.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    provenance = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    score_shift = bool(by[TEST_SHIFT])
    if score_shift and prereg_path:
        provenance["prereg"] = require_lock(prereg_path, lock_path, override_prereg)
    elif score_shift:
        provenance["prereg"] = {"status": "no_prereg_path_given"}

    from ..runtime import log
    log(f"evaluating {meta.name} on {domain}: {len(by[VALID])} validation, "
        f"{len(by[TEST_ID])} in-domain test, {len(by[TEST_SHIFT])} shifted test items")
    log("[1/4] predictions: validation"); frames = [_predict_split(model, by[VALID], VALID)]
    log("[1/4] predictions: in-domain test"); frames.append(_predict_split(model, by[TEST_ID], TEST_ID))
    if score_shift:
        log("[1/4] predictions: shifted test")
        frames.append(_predict_split(model, by[TEST_SHIFT], TEST_SHIFT))
    preds = pd.concat([f for f in frames if len(f)], ignore_index=True)
    preds.insert(0, "model", meta.name)
    preds.to_csv(out / "preds.csv", index=False)

    log("[2/4] record-dependence probes (closure, deletion, control, substitution)")
    probe = probe_model(model, by[TEST_ID], probe_cfg, verifier, filler_items=by[TRAIN] or by[TEST_ID])
    probe.to_csv(out / "probe.csv", index=False)
    chance = chance_rates(by[TRAIN]) if by[TRAIN] else None
    summ = summarize_probe(probe, chance, n_boot=n_boot, correction=probe_cfg.control_correction)
    summ_pol = summarize_probe(probe, chance, n_boot=0, by="polarity",
                                correction=probe_cfg.control_correction)

    v = preds[preds.split == VALID]
    s = preds[preds.split == TEST_SHIFT]
    rec = {**asdict(meta), "domain": domain,
           "acc_val": float(v.correct.mean()) if len(v) else np.nan,
           "acc_id": point(summ, "accuracy"),
           "acc_shift": float(s.correct.mean()) if len(s) else np.nan,
           "n_id": int(point(summ, "n_items")), "n_shift": int(len(s))}
    for k in ("RD", "RD_raw", "RD_diff", "RD_xor", "RD_ctrl_change", "SUF", "TR", "UC", "UC_chance", "AB", "n_rd", "n_tr",
              "TR_kept_original", "TR_abstained", "TR_followed_partly", "TR_other"):
        rec[k] = point(summ, k)
        row = summ[(summ.metric == k) & (summ.stratum == "all")]
        if len(row) and "lo" in row:
            rec[f"{k}_lo"], rec[f"{k}_hi"] = float(row["lo"].iloc[0]), float(row["hi"].iloc[0])
    rec["by_polarity"] = {r.stratum: {} for r in summ_pol.itertuples()}
    for r in summ_pol.itertuples():
        rec["by_polarity"][r.stratum][r.metric] = r.estimate
    if len(v) and len(s):
        rec["atc_pred"] = atc_predict(v.confidence.values, v.correct.values, s.confidence.values)
        rec["doc_pred"] = doc_predict(v.confidence.values, v.correct.values, s.confidence.values)
    if contextcite:
        log("[3/4] ContextCite baseline (32 masked passes per item; the slowest step)")
        rec.update(contextcite_dependence(model, by[TEST_ID], seed=meta.seed))
    else:
        log("[3/4] ContextCite baseline skipped (--no-contextcite)")
    if ablations:
        log("[4/4] measurement-ablation arms")
        abl_items = by[TEST_ID][:ablation_max_items] if ablation_max_items else by[TEST_ID]
        arms = [a for a in default_arms(probe_cfg) if not (a.name == "contextcite" and contextcite)]
        res = run_measurement_ablations(model, abl_items, by[TRAIN], verifier, alt_verifier, arms,
                                        seed=meta.seed)
        rec.update({f"abl_{k}": v for k, v in res.scores.items()})
        if contextcite:
            rec["abl_contextcite"] = rec.get("cc_dependence")
    rec["cache_hit_rate"] = model.hits / model.calls if model.calls else 0.0
    rec["provenance"] = provenance
    (out / "summary.json").write_text(json.dumps(rec, indent=2, default=float))
    log(f"done: acc_id={rec.get('acc_id'):.3f}  RD={rec.get('RD'):.3f}  -> {out / 'summary.json'}")
    return rec

"""Agreement-on-the-line (Baek et al., NeurIPS 2022).

For every pair of models, agreement (share of items where they give the same
answer) is computed in-domain and on the unlabelled target. A line is fitted
to probit(target agreement) vs probit(ID agreement); because its slope and
bias match those of the accuracy line when accuracy-on-the-line holds, each
model's target accuracy is predicted as

    Phi( a * Phi^-1(ID accuracy) + b ).

Only unlabelled target predictions are used.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..normalize import normalize_text


def _probit(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    return norm.ppf(np.clip(x, eps, 1 - eps))


def agreement_on_the_line(preds: pd.DataFrame, id_split: str = "test_id",
                          tgt_split: str = "test_shift") -> pd.DataFrame:
    """``preds`` columns: model, split, item_id, pred, correct.

    Returns one row per model: aotl_pred (predicted target accuracy), plus the
    fitted slope / bias and the agreement-line R^2 for diagnostics.
    """
    p = preds.assign(norm_pred=preds["pred"].astype(str).map(normalize_text))
    piv = {s: p[p.split == s].pivot_table(index="item_id", columns="model", values="norm_pred",
                                          aggfunc="first") for s in (id_split, tgt_split)}
    models = sorted(set(piv[id_split].columns) & set(piv[tgt_split].columns))
    xs, ys = [], []
    for a, b in combinations(models, 2):
        for s, store in ((id_split, xs), (tgt_split, ys)):
            sub = piv[s][[a, b]].dropna()
            store.append(float((sub[a] == sub[b]).mean()) if len(sub) else np.nan)
    xs, ys = np.array(xs), np.array(ys)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return pd.DataFrame({"model": models, "aotl_pred": np.nan})
    X, Y = _probit(xs[ok]), _probit(ys[ok])
    slope, bias = np.polyfit(X, Y, 1)
    r2 = float(np.corrcoef(X, Y)[0, 1] ** 2)
    acc_id = p[p.split == id_split].groupby("model")["correct"].mean()
    out = []
    for m in models:
        out.append({"model": m, "aotl_pred": float(norm.cdf(slope * _probit(np.array([acc_id[m]]))[0] + bias)),
                    "aotl_slope": slope, "aotl_bias": bias, "aotl_r2": r2})
    return pd.DataFrame(out)

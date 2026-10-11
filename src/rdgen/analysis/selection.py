"""Model-selection regret and the interventional test (methodology 10.5-10.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

DEFAULT_CRITERIA = ("acc_id", "RD", "combined", "aotl_pred", "atc_pred", "doc_pred", "cc_dependence")


def add_combined(table: pd.DataFrame) -> pd.DataFrame:
    """Combined criterion = z(acc_id) + z(RD) within domain (no fitting, so no
    in-sample leakage from the shifted labels)."""
    def z(s):
        return (s - s.mean()) / (s.std(ddof=0) or 1.0)
    out = table.copy()
    out["combined"] = out.groupby("domain", group_keys=False).apply(
        lambda g: z(g["acc_id"]) + z(g["RD"].fillna(g["RD"].mean())))
    return out


def regret(table: pd.DataFrame, criteria=DEFAULT_CRITERIA) -> pd.DataFrame:
    t = add_combined(table)
    rows = []
    for d, g in t.groupby("domain"):
        best = g["acc_shift"].max()
        for c in criteria:
            if c not in g or g[c].isna().all():
                continue
            gg = g.dropna(subset=[c])
            pick = gg.loc[gg[c].idxmax()]
            rho = spearmanr(gg[c], gg["acc_shift"]).statistic if len(gg) > 2 else np.nan
            rows.append({"domain": d, "criterion": c, "picked": pick["name"],
                         "regret": float(best - pick["acc_shift"]), "spearman": float(rho)})
    return pd.DataFrame(rows)


def match_pairs(g: pd.DataFrame, delta: float = 0.01, min_rd_gap: float = 0.05,
                same: tuple[str, ...] = ("base_model",)) -> pd.DataFrame:
    """Greedy matching: within strata of ``same``, pair models whose in-domain
    accuracies differ by <= delta, preferring the largest RD gap; each model is
    used at most once."""
    pairs = []
    keys = [k for k in same if k in g]
    strata = g.groupby(keys) if keys else [((), g)]
    for _, s in strata:
        s = s.reset_index(drop=True)
        cand = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                a, b = s.iloc[i], s.iloc[j]
                if abs(a.acc_id - b.acc_id) <= delta and abs(a.RD - b.RD) >= min_rd_gap:
                    cand.append((abs(a.RD - b.RD), i, j))
        used = set()
        for gap, i, j in sorted(cand, reverse=True):
            if i in used or j in used:
                continue
            used |= {i, j}
            hi, lo = (s.iloc[i], s.iloc[j]) if s.iloc[i].RD > s.iloc[j].RD else (s.iloc[j], s.iloc[i])
            pairs.append({"hi_rd": hi["name"], "lo_rd": lo["name"], "family": hi.get("family", ""),
                          "rd_gap": hi.RD - lo.RD, "acc_id_gap": hi.acc_id - lo.acc_id,
                          "shift_gain": hi.acc_shift - lo.acc_shift})
    return pd.DataFrame(pairs)


def interventional(table: pd.DataFrame, delta: float = 0.01, min_rd_gap: float = 0.05,
                   same: tuple[str, ...] = ("base_model",), n_boot: int = 2000, seed: int = 0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_pairs, rows = [], []
    for d, g in table.dropna(subset=["acc_id", "RD", "acc_shift"]).groupby("domain"):
        p = match_pairs(g, delta, min_rd_gap, same)
        if p.empty:
            rows.append({"domain": d, "n_pairs": 0})
            continue
        p["domain"] = d
        all_pairs.append(p)
        gains = p["shift_gain"].to_numpy()
        boots = [rng.choice(gains, len(gains), replace=True).mean() for _ in range(n_boot)]
        k = int((gains > 0).sum())
        rows.append({"domain": d, "n_pairs": len(p), "tau": float(gains.mean()),
                     "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                     "sign_p_one_sided": float(binomtest(k, len(gains), 0.5, alternative="greater").pvalue)})
    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    return pd.DataFrame(rows), pairs

"""Ablation analysis (protocol step 6) and the gate test (H2).

Measurement ablations
---------------------
Every ``abl_<arm>`` column in the population table is one way of measuring
record-dependence. For each arm the pre-registered prediction analysis is
re-run with that arm as the predictor. Predictors are z-scored within domain
first, so effects are "per standard deviation" and comparable across arms
whose raw scales differ (RD in [0, 1], ContextCite shares, etc.).

Reported per arm:
  validity      Spearman with ground-truth reliance (synthetic/mock runs only)
  agreement     Spearman with the primary arm
  beta_sd       pooled effect per SD, 95% CI, one-sided p, domains positive
  dR2           mean gain in explained variance over in-domain accuracy alone
  lofo_gain     mean leave-one-family-out RMSE reduction (out-of-sample)
  regret        model-selection regret of z(acc_id) + z(arm)

Robustness subgroups (primary arm): by context strategy, trained models only,
and models never trained with a gate or abstention augmentation (Goodhart).

Gate ablation (H2)
------------------
Each gated model is compared with the sft model of the same base model, seed
and domain. Per gate mode: mean shifted-accuracy gain, share of the ungated
gain retained, and the change in RD, with cluster-bootstrap CIs over base
models. H2 holds when the pre-registered modes retain at least
``retained_min`` of the ungated gain and RD is non-inferior to sft
(lower CI bound of the RD change above ``-rd_margin``).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .fit import fit_all, success

ARM_ORDER = ["primary", "no_control", "control_difference", "control_xor", "closure_top1",
             "closure_gold", "del_explicit_neg", "del_all_pol", "substitution", "sufficiency",
             "verifier_alt", "contextcite"]
ARM_LABEL = {
    "primary": "Primary (full closure, union control, pos+span)",
    "no_control": "No deletion control",
    "control_difference": "Control by subtraction",
    "control_xor": "Control as cancelling flips",
    "closure_top1": "Delete first evidence unit only",
    "closure_gold": "Gold evidence only (no verifier)",
    "del_explicit_neg": "Deletion also on explicit negatives",
    "del_all_pol": "Deletion on all polarities",
    "substitution": "Substitution estimator (TR)",
    "sufficiency": "Sufficiency (SUF)",
    "verifier_alt": "Alternative verifier",
    "contextcite": "ContextCite-style (no gold evidence)",
}


def _z_within(df: pd.DataFrame, col: str) -> pd.Series:
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0
    return df.groupby("domain", group_keys=False)[col].apply(z)


def _mean_spearman(df: pd.DataFrame, a: str, b: str) -> float:
    vals = []
    for _, g in df.dropna(subset=[a, b]).groupby("domain"):
        if len(g) > 3 and g[a].nunique() > 1 and g[b].nunique() > 1:
            vals.append(spearmanr(g[a], g[b]).statistic)
    return float(np.mean(vals)) if vals else np.nan


def _regret(df: pd.DataFrame, col: str) -> float:
    out = []
    for _, g in df.dropna(subset=[col, "acc_id", "acc_shift"]).groupby("domain"):
        if len(g) < 2:
            continue
        crit = _z_within(g, "acc_id") + _z_within(g, col)
        out.append(g["acc_shift"].max() - g.loc[crit.idxmax(), "acc_shift"])
    return float(np.mean(out)) if out else np.nan


def measurement_ablation(table: pd.DataFrame, prereg: dict) -> pd.DataFrame:
    h = prereg["analysis"]
    covs = h.get("covariates", [])
    arms = [a for a in ARM_ORDER if f"abl_{a}" in table and table[f"abl_{a}"].notna().sum() >= 5]
    truth = "x_true_reliance" if "x_true_reliance" in table else None
    rows = []
    for arm in arms:
        col = f"abl_{arm}"
        t = table.dropna(subset=[col]).copy()
        if t.groupby("domain")[col].std(ddof=0).fillna(0).max() < 1e-9:
            rows.append({"arm": arm, "label": ARM_LABEL.get(arm, arm),
                         "note": "degenerate: no variance across models", "domains_pos": 0,
                         "n_domains": int(t["domain"].nunique()), "H1_rule": False})
            continue
        t["_pred"] = _z_within(t, col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fd, pooled = fit_all(t, "_pred", covs)
        verdict = success(fd, pooled, h["alpha"], h["alpha_domain"], h["k_domains_required"]) if len(fd) else {}
        rows.append({
            "arm": arm, "label": ARM_LABEL.get(arm, arm),
            "validity": _mean_spearman(t, col, truth) if truth else np.nan,
            "agreement": _mean_spearman(t, col, "abl_primary") if "abl_primary" in t else np.nan,
            "beta_sd": pooled.get("pooled", np.nan), "lo": pooled.get("lo", np.nan),
            "hi": pooled.get("hi", np.nan), "p": pooled.get("p_one_sided", np.nan),
            "domains_pos": verdict.get("n_domains_positive", 0), "n_domains": verdict.get("n_domains", 0),
            "H1_rule": verdict.get("H1_supported", False),
            "dR2": float((fd["r2_with"] - fd["r2_without"]).mean()) if len(fd) else np.nan,
            "lofo_gain": float((fd["lofo_rmse_without"] - fd["lofo_rmse_with"]).mean()) if len(fd) else np.nan,
            "regret": _regret(t, col), "note": "",
        })
    return pd.DataFrame(rows)


def robustness_subgroups(table: pd.DataFrame, prereg: dict, predictor: str = "RD") -> pd.DataFrame:
    h = prereg["analysis"]
    covs = [c for c in h.get("covariates", []) if c != "context"]
    groups = {"all models": table}
    if "regime" in table:
        groups["trained only (no zero-shot)"] = table[table.regime != "zeroshot"]
        groups["never gated or abstention-trained"] = table[~table.regime.isin(["gated", "abstain_aug"])]
    if "context" in table and table["context"].nunique() > 1:
        for c, g in table.groupby("context"):
            groups[f"context = {c}"] = g
    rows = []
    for name, g in groups.items():
        if g[predictor].notna().sum() < 10:
            rows.append({"subgroup": name, "n_models": len(g)})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fd, pooled = fit_all(g, predictor, covs)
        rows.append({"subgroup": name, "n_models": len(g), "beta": pooled.get("pooled", np.nan),
                     "lo": pooled.get("lo", np.nan), "hi": pooled.get("hi", np.nan),
                     "p": pooled.get("p_one_sided", np.nan),
                     "domains_pos": success(fd, pooled, h["alpha"], h["alpha_domain"],
                                            h["k_domains_required"])["n_domains_positive"]})
    return pd.DataFrame(rows)


def gate_h2(table: pd.DataFrame, prereg: dict, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    cfg = prereg.get("h2", {})
    retained_min = cfg.get("retained_min", 0.7)
    margin = cfg.get("rd_margin", 0.02)
    test_modes = cfg.get("modes_tested", ["necessity", "both"])
    if "x_gate_mode" not in table or "regime" not in table:
        return pd.DataFrame(), {"status": "no gate-ablation runs in table"}
    keys = ["domain", "base_model", "seed"]
    sft = table[table.regime == "sft"].set_index(keys)
    gated = table[(table.regime == "gated") & table.x_gate_mode.notna()]
    pairs = []
    for _, r in gated.iterrows():
        k = (r.domain, r.base_model, r.seed)
        if k not in sft.index:
            continue
        b = sft.loc[k]
        b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
        pairs.append({"domain": r.domain, "base_model": r.base_model, "seed": r.seed,
                      "mode": r.x_gate_mode, "gain": r.acc_shift - b.acc_shift,
                      "d_rd": r.RD - b.RD, "d_acc_id": r.acc_id - b.acc_id})
    p = pd.DataFrame(pairs)
    if p.empty:
        return p, {"status": "no gated/sft pairs"}
    # H2 is tested on a difference, not a ratio: D = gain(mode) - retained_min * gain(none).
    # A retained-share ratio is unstable when the ungated gain is near zero.
    rng = np.random.default_rng(seed)
    clusters = p["base_model"].unique()
    boot = {m: {"gain": [], "d_rd": [], "D": []} for m in p["mode"].unique()}
    for _ in range(n_boot):
        pick = rng.choice(clusters, len(clusters), replace=True)
        b = pd.concat([p[p.base_model == c] for c in pick])
        g_none_b = b[b["mode"] == "none"]["gain"].mean()
        for m, gm in b.groupby("mode"):
            boot[m]["gain"].append(gm["gain"].mean())
            boot[m]["d_rd"].append(gm["d_rd"].mean())
            boot[m]["D"].append(gm["gain"].mean() - retained_min * g_none_b)
    g_none = p[p["mode"] == "none"]["gain"].mean()
    unstable = not np.isfinite(g_none) or abs(g_none) < cfg.get("min_ungated_gain", 0.01)

    def ci(v):
        v = np.asarray(v, float)
        return np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)

    rows = []
    for m, gm in p.groupby("mode"):
        gl, gh = ci(boot[m]["gain"])
        rl, rh = ci(boot[m]["d_rd"])
        dl, dh = ci(boot[m]["D"])
        rows.append({"mode": m, "n_pairs": len(gm), "gain": gm["gain"].mean(), "gain_lo": gl, "gain_hi": gh,
                     "retained": np.nan if unstable else gm["gain"].mean() / g_none,
                     "D": gm["gain"].mean() - retained_min * g_none, "D_lo": dl, "D_hi": dh,
                     "d_rd": gm["d_rd"].mean(), "d_rd_lo": rl, "d_rd_hi": rh,
                     "d_acc_id": gm["d_acc_id"].mean()})
    res = pd.DataFrame(rows).set_index("mode").reindex(["none", "entailment", "necessity", "both"]).dropna(
        how="all").reset_index()
    verdict = {"retained_min": retained_min, "rd_margin": margin, "test": "D_lo > 0 and d_rd_lo > -margin",
               "ungated_gain": float(g_none), "modes": {}}
    for m in test_modes:
        r = res[res["mode"] == m]
        if r.empty:
            continue
        r = r.iloc[0]
        ok = bool(r.D_lo > 0 and r.d_rd_lo > -margin)
        verdict["modes"][m] = {"D": float(r.D), "D_lo": float(r.D_lo), "d_rd": float(r.d_rd),
                               "d_rd_lo": float(r.d_rd_lo), "passes": ok}
    verdict["H2_supported"] = bool(verdict["modes"]) and any(v["passes"] for v in verdict["modes"].values())
    if unstable:
        verdict["note"] = ("ungated gain is near zero, so the retained share is not reported; "
                           "the difference test D still applies")
    return res, verdict


def to_latex(df: pd.DataFrame, cols: list[tuple[str, str, str]], caption: str, label: str) -> str:
    """Booktabs table. ``cols`` = (column, header, format)."""
    head = " & ".join(h for _, h, _ in cols)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule", head + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c, _, fmt in cols:
            v = row.get(c)
            if fmt == "s":
                cells.append(str(v).replace("_", r"\_").replace("&", r"\&"))
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("--")
            else:
                cells.append(format(v, fmt))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def run_ablation_analysis(table: pd.DataFrame, prereg: dict, out_dir: str | Path) -> dict:
    from .figures import fig_ablation
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    meas = measurement_ablation(table, prereg)
    if len(meas):
        for c in ("validity", "agreement", "beta_sd", "lo", "hi", "p", "dR2", "lofo_gain", "regret"):
            if c not in meas:
                meas[c] = np.nan
        meas["ci"] = meas.apply(lambda r: f"[{r.lo:.2f}, {r.hi:.2f}]" if pd.notna(r.lo) else r.note, axis=1)
        meas.to_csv(out / "table2_measurement_ablation.csv", index=False)
        (out / "table2_measurement_ablation.tex").write_text(to_latex(
            meas, [("label", "Estimator variant", "s"), ("validity", "Validity", ".2f"),
                   ("agreement", "Agr.", ".2f"), ("beta_sd", r"$\beta$/SD", ".3f"), ("ci", "95\\% CI", "s"),
                   ("domains_pos", "Dom.+", "d"), ("dR2", r"$\Delta R^2$", ".3f"),
                   ("lofo_gain", "LOFO gain", ".3f"), ("regret", "Regret", ".3f")],
            "Measurement ablations. Each row re-runs the pre-registered prediction analysis with one "
            "estimator variant as the predictor (z-scored within domain). Validity: Spearman with "
            "ground-truth reliance where known. Agr.: Spearman with the primary estimator. "
            "Dom.+: domains with a significant positive effect.", "tab:ablation"))
        for ext in ("pdf", "png"):
            fig_ablation(meas, out / f"fig_ablation.{ext}")
        result["measurement"] = meas.drop(columns=["ci"]).to_dict("records")
    rob = robustness_subgroups(table, prereg)
    rob.to_csv(out / "robustness_subgroups.csv", index=False)
    result["robustness"] = rob.to_dict("records")
    h2_tab, h2 = gate_h2(table, prereg)
    if len(h2_tab):
        h2_tab.to_csv(out / "table3_gate_ablation.csv", index=False)
        (out / "table3_gate_ablation.tex").write_text(to_latex(
            h2_tab, [("mode", "Gate", "s"), ("n_pairs", "Pairs", "d"), ("gain", "Shift gain", ".3f"),
                     ("D", "$D$", ".3f"), ("D_lo", "$D$ low", ".3f"), ("d_rd", r"$\Delta$RD", ".3f"),
                     ("d_rd_lo", r"$\Delta$RD low", ".3f"), ("d_acc_id", r"$\Delta$ID acc", ".3f")],
            "Gate ablation (H2). Each gated model is compared with the fine-tuned model of the same base "
            "model, seed and domain. $D$ = shifted-accuracy gain minus 0.7 times the ungated gain; "
            "H2 requires the lower 95\\% bound of $D$ above 0 and of $\\Delta$RD above $-0.02$.",
            "tab:gate"))
    result["H2"] = h2
    (out / "ablation_results.json").write_text(json.dumps(result, indent=2, default=float))
    _write_md(out, meas, rob, h2_tab, h2)
    return result


def _write_md(out: Path, meas, rob, h2_tab, h2) -> None:
    lines = ["# Ablation report", "", "## Measurement ablations (Table 2)", ""]
    if len(meas):
        lines.append(meas[["label", "validity", "agreement", "beta_sd", "ci", "domains_pos", "dR2",
                           "lofo_gain", "regret"]].round(3).to_markdown(index=False))
    lines += ["", "## Robustness of the primary estimator", "", rob.round(3).to_markdown(index=False),
              "", "## Gate ablation (H2, Table 3)", ""]
    if len(h2_tab):
        lines.append(h2_tab.round(3).to_markdown(index=False))
    lines += ["", f"**H2 supported:** {h2.get('H2_supported', 'n/a')}  ",
              f"{json.dumps(h2.get('modes', {}), default=float)}"]
    if "note" in h2:
        lines.append(f"\nNote: {h2['note']}")
    (out / "ablation_report.md").write_text("\n".join(lines) + "\n")

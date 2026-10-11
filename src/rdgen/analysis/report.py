"""Run the full pre-registered analysis on a population table and write a
report: tables (CSV), figures (PDF + PNG) and a markdown summary that states
whether the pre-registered success rule is met."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import yaml

from .figures import fig_forest, fig_regret, fig_scatter
from .fit import family_bootstrap_pooled, fit_all, success
from .selection import interventional, regret


def run_analysis(table: pd.DataFrame, prereg: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = prereg["analysis"]
    covs = h.get("covariates", [])
    predictor = h.get("primary_predictor", "RD")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_df, pooled = fit_all(table, predictor, covs)
        boot = family_bootstrap_pooled(table, predictor, covs, n_boot=h.get("n_boot_family", 200))
    pooled.update(boot)
    verdict = success(fit_df, pooled, h["alpha"], h["alpha_domain"], h["k_domains_required"])
    reg = regret(table, h.get("criteria", None) or
                 ("acc_id", "RD", "combined", "aotl_pred", "atc_pred", "doc_pred", "cc_dependence"))
    iv = prereg["interventional"]
    iv_sum, pairs = interventional(table, iv["delta_acc_id"], iv["min_rd_gap"], tuple(iv["match_on"]))

    # secondary: same fits with the alternative estimators (ablations)
    secondary = {}
    for alt in h.get("secondary_predictors", []):
        if alt in table and table[alt].notna().sum() >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fd2, p2 = fit_all(table, alt, covs)
            secondary[alt] = {"pooled": p2, "per_domain": fd2.to_dict("records")}

    fit_df.to_csv(out / "per_domain_fits.csv", index=False)
    reg.to_csv(out / "regret.csv", index=False)
    iv_sum.to_csv(out / "interventional.csv", index=False)
    pairs.to_csv(out / "interventional_pairs.csv", index=False)
    for ext in ("pdf", "png"):
        fig_scatter(table, out / f"fig2_scatter.{ext}")
        if len(fit_df):
            fig_forest(fit_df, pooled, out / f"fig3_forest.{ext}")
        if len(reg):
            fig_regret(reg, out / f"fig4_regret.{ext}")

    result = {"verdict": verdict, "pooled": pooled, "interventional": iv_sum.to_dict("records"),
              "secondary": secondary}
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float))
    _write_markdown(out, fit_df, pooled, verdict, reg, iv_sum)
    return result


def _write_markdown(out: Path, fit_df, pooled, verdict, reg, iv_sum) -> None:
    lines = ["# Pre-registered analysis report", ""]
    lines.append(f"**H1 supported:** {verdict['H1_supported']} "
                 f"(pooled positive: {verdict['pooled_positive']}; "
                 f"domains positive: {verdict['n_domains_positive']}/{verdict['n_domains']}, "
                 f"required {verdict['k_required']})")
    lines.append("")
    lines.append(f"Pooled RD coefficient: {pooled.get('pooled', float('nan')):.3f} "
                 f"[{pooled.get('lo', float('nan')):.3f}, {pooled.get('hi', float('nan')):.3f}], "
                 f"one-sided p = {pooled.get('p_one_sided', float('nan')):.4f}, I^2 = {pooled.get('I2', float('nan')):.2f}; "
                 f"family bootstrap 95% [{pooled.get('boot_lo', float('nan')):.3f}, {pooled.get('boot_hi', float('nan')):.3f}]")
    lines += ["", "## Per-domain fits", "", fit_df.round(4).to_markdown(index=False) if len(fit_df) else "_none_",
              "", "## Model-selection regret (mean over domains)", ""]
    if len(reg):
        lines.append(reg.groupby("criterion")[["regret", "spearman"]].mean().round(4)
                     .sort_values("regret").to_markdown())
    lines += ["", "## Interventional test", "", iv_sum.round(4).to_markdown(index=False) if len(iv_sum) else "_none_"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def load_prereg(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())

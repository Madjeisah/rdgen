"""Paper figures 2-4. Plain matplotlib, colour-blind-safe, one style."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_scatter(table: pd.DataFrame, path: str | Path) -> None:
    """Figure 2: shifted vs in-domain accuracy per domain, coloured by RD."""
    doms = sorted(table["domain"].unique())
    fig, axes = plt.subplots(1, len(doms), figsize=(3.0 * len(doms), 3.0), squeeze=False)
    sc = None
    for ax, d in zip(axes[0], doms):
        g = table[table.domain == d]
        sc = ax.scatter(g.acc_id, g.acc_shift, c=g.RD, cmap="viridis", vmin=0, vmax=1, s=22,
                        edgecolor="none")
        lo = min(g.acc_id.min(), g.acc_shift.min()) - 0.02
        hi = max(g.acc_id.max(), g.acc_shift.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], lw=0.8, ls="--", color="0.6")
        ax.set_title(d)
        ax.set_xlabel("In-domain accuracy")
    axes[0][0].set_ylabel("Shifted accuracy")
    if sc is not None:
        fig.colorbar(sc, ax=axes[0].tolist(), label="Record-dependence (in-domain)", shrink=0.85)
    fig.savefig(path)
    plt.close(fig)


def fig_forest(fit_df: pd.DataFrame, pooled: dict, path: str | Path) -> None:
    """Figure 3: per-domain RD coefficients with 95% CIs and the pooled effect."""
    f = fit_df.sort_values("domain")
    labels = list(f.domain) + ["Pooled (random effects)"]
    est = list(f.beta_rd) + [pooled.get("pooled", np.nan)]
    lo = list(f.beta_rd - 1.96 * f.se_rd) + [pooled.get("lo", np.nan)]
    hi = list(f.beta_rd + 1.96 * f.se_rd) + [pooled.get("hi", np.nan)]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 0.45 * len(labels) + 0.8))
    for yi, e, l, h, lab in zip(y, est, lo, hi, labels):
        pooled_row = lab.startswith("Pooled")
        ax.plot([l, h], [yi, yi], color="black", lw=1.2)
        ax.plot(e, yi, marker="D" if pooled_row else "s", color="#1f6fb4" if pooled_row else "black", ms=6)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("RD coefficient (probit shifted accuracy)")
    fig.savefig(path)
    plt.close(fig)


def fig_ablation(meas: pd.DataFrame, path: str | Path) -> None:
    """Appendix figure: pooled effect per SD for each estimator variant."""
    m = meas.dropna(subset=["beta_sd"]).reset_index(drop=True)
    y = np.arange(len(m))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.36 * len(m) + 0.9))
    for yi, r in zip(y, m.itertuples()):
        primary = r.arm == "primary"
        col = "#1f6fb4" if primary else "black"
        ax.plot([r.lo, r.hi], [yi, yi], color=col, lw=1.2)
        ax.plot(r.beta_sd, yi, marker="D" if primary else "s", color=col, ms=5)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_yticks(y, m["label"])
    ax.set_xlabel("Pooled effect on probit shifted accuracy (per SD of estimator)")
    fig.savefig(path)
    plt.close(fig)


def fig_regret(reg: pd.DataFrame, path: str | Path) -> None:
    """Figure 4: mean model-selection regret per criterion across domains."""
    m = reg.groupby("criterion")["regret"].agg(["mean", "sem"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(4.8, 0.35 * len(m) + 0.8))
    colors = ["#1f6fb4" if c in ("RD", "combined") else "0.6" for c in m.index]
    ax.barh(m.index, m["mean"], xerr=m["sem"].fillna(0), color=colors)
    ax.set_xlabel("Regret (shifted accuracy lost vs best model)")
    fig.savefig(path)
    plt.close(fig)

"""The prediction model (methodology 10.5) and its pooling across domains.

Per domain:
    probit(acc_shift) ~ probit(acc_id) + RD + covariates  [+ family effect]

Primary per-domain fit: a linear mixed model with a random intercept per model
family (models from one family are not independent). If it fails to converge
or there are too few families, OLS with family-clustered standard errors is
used and the fallback is recorded.

Pooling: DerSimonian-Laird random-effects meta-analysis of the per-domain RD
coefficients (forest plot), plus a family-cluster bootstrap of the pooled
estimate. Out-of-sample check: leave-one-family-out (LOFO) prediction error
with and without RD.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, t as student_t


def probit(x, eps: float = 1e-3):
    return norm.ppf(np.clip(np.asarray(x, float), eps, 1 - eps))


def prepare(table: pd.DataFrame, predictor: str = "RD") -> pd.DataFrame:
    df = table.dropna(subset=["acc_id", "acc_shift", predictor]).copy()
    df["y"] = probit(df["acc_shift"])
    df["x_acc"] = probit(df["acc_id"])
    df["x_rd"] = df[predictor].astype(float)
    if "size_b" in df:
        df["log_size"] = np.log(df["size_b"].astype(float).clip(lower=0.1)).fillna(0.0)
    return df


def _formula(df: pd.DataFrame, covariates: list[str], with_rd: bool = True) -> str:
    terms = ["x_acc"] + (["x_rd"] if with_rd else [])
    for c in covariates:
        if c in df and df[c].nunique() > 1:
            terms.append(f"C({c})" if df[c].dtype == object else c)
    return "y ~ " + " + ".join(terms)


@dataclass
class DomainFit:
    domain: str
    beta_rd: float
    se_rd: float
    p_one_sided: float
    beta_acc: float
    r2_without: float
    r2_with: float
    lofo_rmse_without: float
    lofo_rmse_with: float
    n_models: int
    n_families: int
    method: str


def _lofo(df: pd.DataFrame, covariates: list[str], with_rd: bool) -> float:
    errs = []
    for fam in df["family"].unique():
        tr, te = df[df.family != fam], df[df.family == fam]
        if len(tr) < 4:
            continue
        f = _formula(tr, [c for c in covariates if c in tr and tr[c].nunique() > 1
                          and set(te[c]).issubset(set(tr[c]))], with_rd)
        try:
            m = smf.ols(f, tr).fit()
            pred = norm.cdf(m.predict(te))
            errs += list((pred - te["acc_shift"].values) ** 2)
        except Exception:
            continue
    return float(np.sqrt(np.mean(errs))) if errs else np.nan


def fit_domain(df: pd.DataFrame, domain: str, covariates: list[str] | None = None,
               min_families_mixed: int = 4) -> DomainFit:
    covariates = covariates or []
    f_with, f_without = _formula(df, covariates, True), _formula(df, covariates, False)
    n_fam = df["family"].nunique()
    method = "mixedlm"
    beta = se = np.nan
    if n_fam >= min_families_mixed:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(f_with, df, groups=df["family"]).fit(reml=True)
            if not m.converged:
                raise RuntimeError("no convergence")
            beta, se = float(m.params["x_rd"]), float(m.bse["x_rd"])
            dof = max(n_fam - 2, 1)
        except Exception:
            method = "ols_cluster"
    else:
        method = "ols_cluster"
    if method == "ols_cluster":
        m = smf.ols(f_with, df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["family"])[0]}) \
            if n_fam > 1 else smf.ols(f_with, df).fit()
        beta, se = float(m.params["x_rd"]), float(m.bse["x_rd"])
        dof = max(n_fam - 1, 1)
    p = float(1 - student_t.cdf(beta / se, dof)) if se > 0 else np.nan
    r2_w = float(smf.ols(f_with, df).fit().rsquared)
    r2_wo = float(smf.ols(f_without, df).fit().rsquared)
    m_acc = smf.ols(f_with, df).fit()
    return DomainFit(domain, beta, se, p, float(m_acc.params["x_acc"]), r2_wo, r2_w,
                     _lofo(df, covariates, False), _lofo(df, covariates, True),
                     len(df), n_fam, method)


def random_effects(betas: np.ndarray, ses: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling."""
    b, s = np.asarray(betas, float), np.asarray(ses, float)
    ok = ~(np.isnan(b) | np.isnan(s) | (s <= 0))
    b, s = b[ok], s[ok]
    k = len(b)
    if k == 0:
        return {"pooled": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan, "p_one_sided": np.nan,
                "tau2": np.nan, "I2": np.nan, "k": 0}
    w = 1 / s ** 2
    fixed = np.sum(w * b) / np.sum(w)
    Q = float(np.sum(w * (b - fixed) ** 2))
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 and k > 1 else 0.0
    w_re = 1 / (s ** 2 + tau2)
    pooled = float(np.sum(w_re * b) / np.sum(w_re))
    se = float(np.sqrt(1 / np.sum(w_re)))
    I2 = float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0
    return {"pooled": pooled, "se": se, "lo": pooled - 1.96 * se, "hi": pooled + 1.96 * se,
            "p_one_sided": float(1 - norm.cdf(pooled / se)), "tau2": tau2, "I2": I2, "k": k}


def fit_all(table: pd.DataFrame, predictor: str = "RD", covariates: list[str] | None = None
            ) -> tuple[pd.DataFrame, dict]:
    df = prepare(table, predictor)
    fits = [fit_domain(g, d, covariates) for d, g in df.groupby("domain") if len(g) >= 5]
    fit_df = pd.DataFrame([f.__dict__ for f in fits])
    pooled = random_effects(fit_df["beta_rd"].values, fit_df["se_rd"].values) if len(fit_df) else {}
    return fit_df, pooled


def family_bootstrap_pooled(table: pd.DataFrame, predictor: str = "RD", covariates=None,
                            n_boot: int = 200, seed: int = 0) -> dict:
    """Resample model families within each domain; re-fit and re-pool."""
    rng = np.random.default_rng(seed)
    df = table.copy()
    est = []
    for _ in range(n_boot):
        parts = []
        for d, g in df.groupby("domain"):
            fams = g["family"].unique()
            pick = rng.choice(fams, size=len(fams), replace=True)
            for j, f in enumerate(pick):
                parts.append(g[g.family == f].assign(family=f"{f}#{j}"))
        boot = pd.concat(parts, ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fd, pooled = fit_all(boot, predictor, covariates)
            if pooled and not np.isnan(pooled["pooled"]):
                est.append(pooled["pooled"])
        except Exception:
            continue
    if not est:
        return {"boot_lo": np.nan, "boot_hi": np.nan, "boot_n": 0}
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {"boot_lo": float(lo), "boot_hi": float(hi), "boot_n": len(est),
            "boot_p_le0": float(np.mean(np.array(est) <= 0))}


def success(fit_df: pd.DataFrame, pooled: dict, alpha: float = 0.05,
            alpha_domain: float = 0.05, k_required: int = 3) -> dict:
    """Pre-registered rule: pooled RD effect > 0 (one-sided) AND a significant
    positive effect in at least ``k_required`` domains."""
    n_pos = int(((fit_df["beta_rd"] > 0) & (fit_df["p_one_sided"] < alpha_domain)).sum()) if len(fit_df) else 0
    pooled_ok = bool(pooled and pooled.get("p_one_sided", 1) < alpha and pooled.get("pooled", 0) > 0)
    return {"pooled_positive": pooled_ok, "n_domains_positive": n_pos,
            "n_domains": int(len(fit_df)), "k_required": k_required,
            "H1_supported": pooled_ok and n_pos >= k_required}

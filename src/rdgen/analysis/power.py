"""Power simulation for the pre-registered success rule (protocol step 7).

Simulates model populations at the model level:

    probit(acc_shift) = b0 + b1 * probit(acc_id) + b2 * RD + u_family + e

with RD and acc_id correlated as they are in practice, plus measurement noise
on RD from finite test sets. Returns the probability that the full analysis
(per-domain fits + random-effects pooling + 'k of D domains' rule) declares
success, as a function of models per domain. Use pilot estimates for the
inputs; the defaults are placeholders, not findings.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fit import fit_all, success


def simulate_population(n_models: int, n_families: int, n_domains: int, beta_rd: float,
                        rng: np.random.Generator, beta_acc: float = 0.9, rd_acc_corr: float = 0.3,
                        sd_family: float = 0.10, sd_noise: float = 0.12, rd_meas_sd: float = 0.04,
                        domain_beta_sd: float = 0.0) -> pd.DataFrame:
    rows = []
    for d in range(n_domains):
        b2 = beta_rd + rng.normal(0, domain_beta_sd)
        fam_eff = rng.normal(0, sd_family, n_families)
        for m in range(n_models):
            fam = m % n_families
            z = rng.normal(size=2)
            rd_true = norm.cdf(0.3 + 0.8 * z[0])
            acc_id = norm.cdf(0.6 + 0.5 * (rd_acc_corr * z[0] + np.sqrt(1 - rd_acc_corr ** 2) * z[1]))
            y = -0.3 + beta_acc * norm.ppf(acc_id) + b2 * rd_true + fam_eff[fam] + rng.normal(0, sd_noise)
            rows.append({"domain": f"d{d}", "name": f"m{m}", "family": f"f{fam}",
                         "acc_id": acc_id, "acc_shift": norm.cdf(y),
                         "RD": float(np.clip(rd_true + rng.normal(0, rd_meas_sd), 0, 1)),
                         "size_b": 1.0})
    return pd.DataFrame(rows)


def power_curve(model_counts=(12, 24, 36, 48, 72), n_families: int = 6, n_domains: int = 4,
                beta_rd: float = 0.5, n_sim: int = 200, k_required: int = 3, seed: int = 0,
                **kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for n in model_counts:
        hits = 0
        for _ in range(n_sim):
            pop = simulate_population(n, n_families, n_domains, beta_rd, rng, **kw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fd, pooled = fit_all(pop)
            hits += success(fd, pooled, k_required=k_required)["H1_supported"]
        out.append({"models_per_domain": n, "beta_rd": beta_rd, "power": hits / n_sim, "n_sim": n_sim})
    return pd.DataFrame(out)

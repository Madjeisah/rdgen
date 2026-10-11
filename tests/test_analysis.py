import numpy as np
import pandas as pd
import pytest

from rdgen.analysis.fit import fit_all, random_effects, success
from rdgen.analysis.power import power_curve, simulate_population
from rdgen.analysis.selection import interventional, regret


def test_random_effects_equal_studies():
    r = random_effects(np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.1, 0.1]))
    assert r["pooled"] == pytest.approx(0.5)
    assert r["tau2"] == pytest.approx(0.0)
    assert r["se"] == pytest.approx(0.1 / np.sqrt(3))


def test_fit_recovers_positive_effect():
    pop = simulate_population(48, 6, 4, beta_rd=0.8, rng=np.random.default_rng(1))
    fd, pooled = fit_all(pop)
    assert pooled["pooled"] > 0.4
    assert success(fd, pooled)["H1_supported"]


def test_fit_null_effect_is_rarely_significant():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(20):
        fd, pooled = fit_all(simulate_population(36, 6, 4, beta_rd=0.0, rng=rng))
        hits += success(fd, pooled)["H1_supported"]
    assert hits <= 2


def test_interventional_detects_gain():
    rows = []
    for d in ("a", "b"):
        for k in range(6):
            base = f"base{k}"
            rows.append({"domain": d, "name": f"{base}-lo", "base_model": base, "family": "f",
                         "acc_id": 0.8, "RD": 0.3, "acc_shift": 0.60})
            rows.append({"domain": d, "name": f"{base}-hi", "base_model": base, "family": "f",
                         "acc_id": 0.805, "RD": 0.7, "acc_shift": 0.70})
    summ, pairs = interventional(pd.DataFrame(rows))
    assert (summ.n_pairs == 6).all()
    assert summ.tau.min() == pytest.approx(0.10)


def test_regret_zero_for_oracle_criterion():
    pop = simulate_population(20, 4, 2, 0.6, np.random.default_rng(3))
    pop["oracle"] = pop["acc_shift"]
    r = regret(pop, ("acc_id", "oracle"))
    assert (r[r.criterion == "oracle"].regret == 0).all()


def test_power_curve_small():
    df = power_curve((24,), n_sim=8, beta_rd=0.8)
    assert df.power.iloc[0] >= 0.5

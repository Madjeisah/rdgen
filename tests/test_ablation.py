import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from rdgen.ablation import CachingAnswerer, run_measurement_ablations
from rdgen.analysis.ablation import gate_h2, measurement_ablation, run_ablation_analysis, to_latex
from rdgen.analysis.power import simulate_population
from rdgen.estimators import correct_for_control
from rdgen.models import MockAnswerer
from rdgen.verify import LexicalVerifier

PREREG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "prereg.yaml").read_text())


def test_cache_dedupes_and_matches_uncached(synth):
    m = MockAnswerer("m", reliance=0.6).fit_prior(synth["all"])
    cm = CachingAnswerer(m)
    items = synth["test_id"][:30]
    a, b = cm.predict(items), cm.predict(items)
    assert [p.answer for p in a] == [p.answer for p in m.predict(items)] == [p.answer for p in b]
    assert cm.hits == len(items)


def test_correction_formulas():
    assert correct_for_control(0.5, 0.0, "union") == pytest.approx(0.5)
    assert correct_for_control(0.6, 0.2, "union") == pytest.approx(0.5)
    assert correct_for_control(0.6, 0.2, "difference") == pytest.approx(0.4)
    assert correct_for_control(0.6, 0.2, "xor") == pytest.approx(0.4 / 0.6)


def _scores(synth, reliance, disruption=0.0):
    m = MockAnswerer(f"m{reliance}-{disruption}", reliance=reliance, disruption=disruption).fit_prior(synth["all"])
    return run_measurement_ablations(m, synth["test_id"], synth["train"], LexicalVerifier()).scores


def test_union_control_recovers_dependence_under_disruption(synth):
    """With disruption, no control inflates guessers and subtraction compresses;
    the union correction stays closest to the disruption-free value."""
    clean = [_scores(synth, r)["primary"] for r in (0.2, 0.9)]
    noisy = [_scores(synth, r, 3.0) for r in (0.2, 0.9)]
    err = {k: sum(abs(n[k] - c) for n, c in zip(noisy, clean))
           for k in ("primary", "no_control", "control_difference")}
    assert err["primary"] < err["no_control"] and err["primary"] < err["control_difference"]


def test_top1_closure_underestimates_with_repeated_evidence(synth):
    s = _scores(synth, 0.95)
    assert s["closure_top1"] < s["primary"] - 0.05


def test_measurement_ablation_flags_degenerate_arm():
    pop = simulate_population(30, 5, 3, 0.8, np.random.default_rng(0))
    pop["abl_primary"] = pop["RD"]
    pop["abl_no_control"] = pop["RD"] + np.random.default_rng(1).normal(0, 0.3, len(pop))
    pop["abl_sufficiency"] = 1.0
    t = measurement_ablation(pop, PREREG).set_index("arm")
    assert t.loc["primary", "beta_sd"] > 0 and t.loc["primary", "agreement"] == pytest.approx(1.0)
    assert t.loc["no_control", "agreement"] < 1.0
    assert "degenerate" in t.loc["sufficiency", "note"]


def _gate_table(gain_none, gain_nec, d_rd_nec):
    rows = []
    for d in ("a", "b"):
        for k in range(8):
            base = dict(domain=d, base_model=f"b{k}", seed=0, acc_id=0.8)
            rows.append({**base, "name": f"b{k}-sft", "regime": "sft", "x_gate_mode": None,
                         "acc_shift": 0.6, "RD": 0.5})
            for mode, g, drd in (("none", gain_none, 0.0), ("necessity", gain_nec, d_rd_nec)):
                rows.append({**base, "name": f"b{k}-{mode}", "regime": "gated", "x_gate_mode": mode,
                             "acc_shift": 0.6 + g + 0.001 * k, "RD": 0.5 + drd})
    return pd.DataFrame(rows)


def test_h2_difference_test():
    res, v = gate_h2(_gate_table(0.10, 0.09, 0.05), PREREG, n_boot=300)
    assert v["modes"]["necessity"]["passes"] and v["H2_supported"]
    res, v = gate_h2(_gate_table(0.10, 0.03, 0.05), PREREG, n_boot=300)    # keeps only 30%
    assert not v["modes"]["necessity"]["passes"]
    res, v = gate_h2(_gate_table(0.10, 0.09, -0.10), PREREG, n_boot=300)   # loses RD
    assert not v["modes"]["necessity"]["passes"]


def test_h2_near_zero_ungated_gain_has_no_ratio():
    res, v = gate_h2(_gate_table(0.0, 0.05, 0.05), PREREG, n_boot=300)
    assert res["retained"].isna().all() and "note" in v
    assert v["modes"]["necessity"]["passes"]          # the difference test still works


def test_latex_is_booktabs():
    tex = to_latex(pd.DataFrame([{"a": "x_y", "b": 0.12345}]), [("a", "A", "s"), ("b", "B", ".2f")], "c", "t")
    assert r"\toprule" in tex and r"x\_y & 0.12" in tex


def test_run_ablation_analysis_writes_outputs(tmp_path):
    pop = simulate_population(24, 4, 3, 0.8, np.random.default_rng(2))
    pop["abl_primary"] = pop["RD"]
    pop["abl_substitution"] = pop["RD"] + np.random.default_rng(3).normal(0, 0.05, len(pop))
    pop["regime"] = "sft"
    run_ablation_analysis(pop, PREREG, tmp_path)
    for f in ("table2_measurement_ablation.tex", "fig_ablation.pdf", "ablation_report.md"):
        assert (tmp_path / f).exists()

import json

import pytest

from rdgen.estimators import ProbeConfig
from rdgen.models import MockAnswerer
from rdgen.pipeline.aggregate import aggregate
from rdgen.pipeline.evaluate import ModelMeta, evaluate_model
from rdgen.pipeline.prereg import PreregError, lock, require_lock


def test_lock_blocks_shift_until_locked_and_detects_edits(tmp_path):
    pre = tmp_path / "prereg.yaml"
    pre.write_text("alpha: 0.05\n")
    lk = tmp_path / "prereg.lock"
    with pytest.raises(PreregError):
        require_lock(pre, lk)
    lock(pre, lk)
    assert require_lock(pre, lk)["status"] == "locked"
    pre.write_text("alpha: 0.10\n")
    with pytest.raises(PreregError):
        require_lock(pre, lk)
    assert require_lock(pre, lk, override=True)["status"] == "override_hash_mismatch"


def test_evaluate_and_aggregate(tmp_path, synth):
    pre = tmp_path / "prereg.yaml"
    pre.write_text("x: 1\n")
    lk = tmp_path / "prereg.lock"
    lock(pre, lk)
    for r in (0.3, 0.6, 0.9):
        m = MockAnswerer(f"mock{r}", family=f"f{int(r * 10)}", reliance=r).fit_prior(synth["all"])
        rec = evaluate_model(m, ModelMeta(m.name, m.family, base_model="b"), synth["all"],
                             tmp_path / "runs", ProbeConfig(), prereg_path=str(pre),
                             lock_path=str(lk), contextcite=False, n_boot=20)
        assert rec["provenance"]["prereg"]["status"] == "locked"
        assert 0 <= rec["RD"] <= 1 and "atc_pred" in rec
    t = aggregate(tmp_path / "runs")
    assert len(t) == 3 and t["aotl_pred"].notna().all()
    assert set(json.loads((tmp_path / "runs/synthetic/mock0.3/summary.json").read_text())) >= {"by_polarity"}

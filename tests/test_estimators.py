from rdgen.estimators import ProbeConfig, chance_rates, point, probe_model, summarize_probe
from rdgen.models import MockAnswerer
from rdgen.verify import LexicalVerifier


def _summary(synth, reliance, cfg=ProbeConfig(), boot=0):
    m = MockAnswerer(f"m{reliance}", reliance=reliance).fit_prior(synth["all"])
    df = probe_model(m, synth["test_id"], cfg, LexicalVerifier(), synth["train"])
    return df, summarize_probe(df, chance_rates(synth["train"]), n_boot=boot)


def test_rd_and_tr_increase_with_reliance(synth):
    rd = [point(_summary(synth, r)[1], "RD") for r in (0.1, 0.5, 0.95)]
    tr = [point(_summary(synth, r)[1], "TR") for r in (0.1, 0.5, 0.95)]
    assert rd[0] < rd[1] < rd[2]
    assert tr[0] < tr[1] < tr[2]
    assert rd[2] > 0.9


def test_control_deletion_does_not_change_mock_answers(synth):
    _, s = _summary(synth, 0.7)
    assert point(s, "RD_ctrl_change") < 0.02


def test_explicit_negatives_are_invisible_to_deletion(synth):
    """Closed-world readers keep answering 'no' when an explicit negation is
    deleted, so deletion-based RD on explicit_neg items is ~0 even for a model
    that always reads. This is why the default estimator excludes them."""
    cfg = ProbeConfig(deletion_polarities=("explicit_neg",))
    df, _ = _summary(synth, 1.0, cfg)
    s = summarize_probe(df, None, n_boot=0, by="polarity")
    assert point(s, "RD", "explicit_neg") < 0.05
    assert point(s, "TR", "explicit_neg") > 0.9


def test_bootstrap_cis_bracket_estimate(synth):
    _, s = _summary(synth, 0.6, boot=200)
    row = s[(s.metric == "RD") & (s.stratum == "all")].iloc[0]
    assert row.lo <= row.estimate <= row.hi

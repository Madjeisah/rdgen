import numpy as np
import pandas as pd

from rdgen.baselines import agreement_on_the_line, atc_predict, contextcite_dependence, doc_predict
from rdgen.models import MockAnswerer
from rdgen.train.gate import apply_gate
from rdgen.train.regimes import build_training_set, counterfactual_abstentions
from rdgen.verify import LexicalVerifier


def test_atc_and_doc():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 1000)
    correct = conf > 0.3                       # perfectly ranked confidences
    assert abs(atc_predict(conf, correct, conf) - correct.mean()) < 0.01
    assert abs(doc_predict(conf, correct, conf - 0.1) - (correct.mean() - 0.1)) < 1e-9


def test_aotl_runs_on_mock_population(synth):
    rows = []
    for r in (0.2, 0.4, 0.6, 0.8):
        m = MockAnswerer(f"m{r}", reliance=r).fit_prior(synth["all"])
        for split, items in (("test_id", synth["test_id"]), ("test_shift", synth["shift"])):
            for it, p in zip(items, m.predict(items)):
                rows.append({"model": m.name, "split": split, "item_id": it.item_id,
                             "pred": p.answer, "correct": p.answer == it.answer})
    out = agreement_on_the_line(pd.DataFrame(rows))
    assert len(out) == 4 and out.aotl_pred.between(0, 1).all()


def test_contextcite_separates_readers_from_guessers(synth):
    lo = contextcite_dependence(MockAnswerer("g", reliance=0.0).fit_prior(synth["all"]),
                                synth["test_id"], n_ablations=16, max_items=40)
    hi = contextcite_dependence(MockAnswerer("r", reliance=1.0).fit_prior(synth["all"]),
                                synth["test_id"], n_ablations=16, max_items=40)
    assert hi["cc_dependence"] > lo["cc_dependence"]


def test_necessity_gate_drops_prior_guessable_items(synth):
    pool = [i for i in synth["train"] if i.polarity in ("pos", "span")][:80]
    guesser = MockAnswerer("probe", reliance=0.0).fit_prior(synth["all"])
    kept, rep = apply_gate(pool, "necessity", LexicalVerifier(), [guesser])
    assert rep.n_fail_necessity > 0 and rep.n_kept == len(kept) < len(pool)
    # every kept item is one the guesser gets wrong without evidence
    for it in kept:
        assert guesser.predict([it])[0].answer != it.answer
    none_kept, _ = apply_gate(pool, "none", None)
    assert len(none_kept) == len(pool)


def test_regimes(synth):
    assert build_training_set(synth["all"], "zeroshot") == []
    aug = build_training_set(synth["all"], "abstain_aug", cf_rate=1.0)
    cf = counterfactual_abstentions(synth["train"], rate=1.0)
    assert len(aug) == len(synth["train"]) + len(cf)
    assert all(c.answer == "not_stated" and c.polarity == "unanswerable" for c in cf)

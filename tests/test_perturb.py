import random

from rdgen.perturb import (control_indices, delete, substitute, sufficiency_item,
                           support_closure)
from rdgen.schema import SPAN_TYPE, Item
from rdgen.verify import LexicalVerifier


def _span_item():
    units = ["Intro text here.", "The value of V0 is gamma.", "Filler one.",
             "Again the value of V0 is gamma.", "Filler two.", "Filler three."]
    return Item("i", "d", "test_id", "g", units, "What is the value of V0?", "gamma",
                SPAN_TYPE, [1], "span", {"template": "V0"})


def test_closure_modes_catch_redundant_mentions():
    it = _span_item()
    assert support_closure(it, "top1") == [1]
    assert support_closure(it, "gold") == [1]
    assert support_closure(it, "full", LexicalVerifier()) == [1, 3]


def test_delete_remaps_evidence():
    it = _span_item()
    d = delete(it, [0, 3], "evidence")
    assert len(d.units) == 4 and d.evidence == [0]
    assert d.meta["deleted"] == [0, 3]


def test_control_is_disjoint_and_size_matched():
    it = _span_item()
    clo = [1, 3]
    c = control_indices(it, clo, random.Random(0))
    assert len(c.indices) == 2 and c.shortfall == 0
    assert not set(c.indices) & set(clo)


def test_sufficiency_preserves_length_and_evidence():
    it = _span_item()
    s = sufficiency_item(it, [1, 3], ["FILL"], random.Random(0))
    assert len(s.units) == len(it.units)
    assert s.units[1] == it.units[1] and s.units[0] == "FILL"


def test_span_substitution_replaces_every_mention():
    it = _span_item()
    sub = substitute(it, [1, 3], ["gamma", "delta"], random.Random(0))
    assert sub.expected == "delta"
    assert all("gamma" not in u for u in sub.item.units)


def test_categorical_substitution_needs_flip(synth):
    pos = next(i for i in synth["test_id"] if i.polarity == "pos")
    sub = substitute(pos, pos.evidence, [], random.Random(0))
    assert sub is not None and sub.expected == "no"
    no_flip = pos.with_units(pos.units, pos.evidence)
    no_flip.meta.pop("flip")
    assert substitute(no_flip, pos.evidence, [], random.Random(0)) is None


def test_substitute_keeps_answer_type():
    from rdgen.perturb import answer_shape, pick_substitute
    rng = random.Random(0)
    pool = ["1066", "Paris", "the blue whale", "12", "Napoleon Bonaparte", "2001"]
    assert answer_shape(pick_substitute("1997", pool, rng)) == "year"
    assert answer_shape(pick_substitute("Barack Obama", pool, rng)) == "name"
    # no alternative of the same type -> no substitution rather than a forced one
    assert pick_substitute("42%", ["Paris", "the blue whale"], rng) is None
    # an alternative already present in the document is not used
    assert pick_substitute("1997", ["1066", "2001"], rng, avoid_text="Built in 1066.") == "2001"


def test_substitution_outcomes_recorded():
    from rdgen.data.synthetic import load_synthetic
    from rdgen.estimators import ProbeConfig, probe_model, summarize_probe, point
    from rdgen.models.mock import MockAnswerer
    items = load_synthetic(n_source=60, n_target=10, seed=1)
    m = MockAnswerer("m", reliance=0.5).fit_prior(items)
    test = [i for i in items if i.split == "test_id"]
    df = probe_model(m, test, ProbeConfig(seed=0))
    assert {"pred_sub", "sub_outcome"} <= set(df.columns)
    s = summarize_probe(df, n_boot=0)
    shares = [point(s, k) for k in ("TR", "TR_kept_original", "TR_abstained", "TR_followed_partly", "TR_other")]
    assert abs(sum(shares) - 1) < 1e-9
    # the mock guesser keeps its prior answer, so 'kept_original' must be present
    assert point(s, "TR_kept_original") > 0

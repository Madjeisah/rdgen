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

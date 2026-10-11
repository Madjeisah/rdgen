import pytest

from rdgen.normalize import is_correct, normalize_text, same_answer, to_label, token_f1
from rdgen.schema import CATEGORICAL_TYPE, SPAN_TYPE, Item, read_jsonl, write_jsonl


@pytest.mark.parametrize("raw,label", [
    ("Yes", "yes"), ("yes.", "yes"), ("Da", "yes"), ("Nu", "no"), ("NK", "not_stated"),
    ("Not stated", "not_stated"), ("Entailment", "yes"), ("Contradiction", "no"),
    ("NotMentioned", "not_stated"), ("no, it is absent", "no"),
])
def test_label_mapping(raw, label):
    assert to_label(raw) == label


def test_span_correctness_and_f1():
    assert is_correct("The Alpha", "alpha", SPAN_TYPE)
    assert not is_correct("beta", "alpha", SPAN_TYPE)
    assert token_f1("stage IIA disease", "stage IIA") == pytest.approx(0.8)
    assert is_correct("stage IIA disease", "stage IIA", SPAN_TYPE, f1_threshold=0.7)
    assert normalize_text("Țară, ăla!") == "tara ala"


def test_same_answer_categorical():
    assert same_answer("Da", "yes", CATEGORICAL_TYPE)
    assert not same_answer("no", "not stated", CATEGORICAL_TYPE)


def test_item_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        Item("x", "d", "train", "g", ["a"], "q", "maybe-ish", CATEGORICAL_TYPE)
    with pytest.raises(ValueError):
        Item("x", "d", "train", "g", ["a"], "q", "yes", CATEGORICAL_TYPE, evidence=[3])
    it = Item("x", "d", "train", "g", ["a", "b"], "q", "yes", CATEGORICAL_TYPE, [1], "pos", {"k": 1})
    write_jsonl([it], tmp_path / "a.jsonl")
    back = read_jsonl(tmp_path / "a.jsonl")[0]
    assert back == it and back.in_rd

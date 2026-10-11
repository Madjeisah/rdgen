"""MedQARo (Romanian oncology epicrises; multilingual clinical case study).

The released format may change, so this loader reads a CSV or JSONL export
with configurable column names. Answers "Da" / "Nu" / "NK" map to yes / no /
not_stated.

Polarity of "Nu" answers cannot be read off the label: it needs to know
whether the epicrisis states the negation or simply omits the topic. Supply a
clinician-annotated ``polarity_file`` (CSV: item_id, polarity, evidence)
produced in WP1. Without it, "Nu" items are marked ``unknown_neg`` and
excluded from record-dependence, with a warning.

Splits: Coltea patients are split 70/15/15 at patient level; Dan Furtuna
patients form the shifted test (S2). ``s1_cancer`` builds the within-centre
cross-cancer split instead (train on breast, test on lung).
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

from ..normalize import to_label
from ..schema import CATEGORICAL_TYPE, NO, NOT_STATED, SPAN_TYPE, TEST_SHIFT, TRAIN, YES, Item
from ..text import split_sentences, stable_id
from . import group_split, register

DEFAULT_COLUMNS = {
    "patient_id": "patient_id", "epicrisis": "epicrisis", "question": "question",
    "answer": "answer", "center": "center", "cancer": "cancer", "qtype": "question_type",
}


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _polarity_table(path: str | None) -> dict[str, tuple[str, list[int]]]:
    if not path:
        return {}
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ev = [int(x) for x in r.get("evidence", "").split(";") if x.strip()]
            out[r["item_id"]] = (r["polarity"], ev)
    return out


@register("medqaro")
def load_medqaro(path: str, columns: dict | None = None, polarity_file: str | None = None,
                 shift: str = "s2", train_center: str = "coltea", seed: int = 13) -> list[Item]:
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    pol_tab = _polarity_table(polarity_file)
    items: list[Item] = []
    n_unknown = 0
    for k, r in enumerate(_rows(Path(path))):
        units = split_sentences(r[cols["epicrisis"]])
        raw = str(r[cols["answer"]]).strip()
        lab = to_label(raw)
        is_cat = lab is not None and len(raw.split()) <= 2
        answer = lab if is_cat else raw
        atype = CATEGORICAL_TYPE if is_cat else SPAN_TYPE
        iid = f"medqaro-{r[cols['patient_id']]}-{k}"
        if iid in pol_tab:
            pol, ev = pol_tab[iid]
        elif answer == YES:
            pol, ev = "pos", []
        elif answer == NO:
            pol, ev = "unknown_neg", []
            n_unknown += 1
        elif answer == NOT_STATED:
            pol, ev = "unanswerable", []
        else:
            pol, ev = "span", [i for i, u in enumerate(units) if raw.lower() in u.lower()]
        center = str(r.get(cols["center"], "")).lower()
        cancer = str(r.get(cols["cancer"], "")).lower()
        if shift == "s2":
            split = TRAIN if train_center in center else TEST_SHIFT
        elif shift == "s1_cancer":
            if train_center not in center:
                continue
            split = TRAIN if "breast" in cancer or "san" in cancer else TEST_SHIFT
        else:
            raise ValueError("shift must be 's2' or 's1_cancer'")
        items.append(Item(
            item_id=iid, domain="medqaro", split=split,
            group_id=stable_id("medqaro", str(r[cols["patient_id"]])), units=units,
            question=r[cols["question"]], answer=answer, answer_type=atype, evidence=ev,
            polarity=pol, meta={"center": center, "cancer": cancer,
                                "qtype": r.get(cols["qtype"], ""), "raw_answer": raw},
        ))
    if n_unknown:
        warnings.warn(f"{n_unknown} 'Nu' answers lack polarity annotation and are excluded "
                      "from record-dependence; supply polarity_file from WP1.")
    return group_split(items, seed=seed)

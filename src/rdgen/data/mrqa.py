"""MRQA 2019 shared task (general domain).

Source sub-datasets (SQuAD, NewsQA, TriviaQA, SearchQA, HotpotQA, NQ) form the
training domain; the out-of-domain dev sets (DROP, DuoRC, RACE,
RelationExtraction, TextbookQA; BioASQ if obtained separately from BioASQ)
form the shifted test. Files are the official
``*.jsonl.gz``: a header line followed by one context per line.

All MRQA answers are spans, so the substitution estimator is preferred here:
deleting the evidence sentence removes the answer string itself.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ..schema import SPAN_TYPE, TEST_SHIFT, TRAIN, Item
from ..text import split_sentences, stable_id
from . import group_split, register


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _evidence_units(context: str, units: list[str], char_spans: list[list[int]]) -> list[int]:
    # Map character spans onto sentence indices by cumulative offsets.
    offsets, cur = [], 0
    for u in units:
        i = context.find(u[: min(len(u), 40)], cur)
        i = cur if i < 0 else i
        offsets.append((i, i + len(u)))
        cur = i + len(u)
    hits = set()
    for s, e in char_spans:
        for k, (a, b) in enumerate(offsets):
            if a <= s < b or a < e <= b or (s <= a and e >= b):
                hits.add(k)
    return sorted(hits)


def _read_file(path: Path, split: str, max_items: int | None) -> list[Item]:
    items: list[Item] = []
    source = path.name.split(".")[0]
    with _open(path) as f:
        for n, line in enumerate(f):
            row = json.loads(line)
            if "header" in row:
                continue
            context = " ".join(row["context"].split())
            units = split_sentences(context)
            if not units:
                continue
            gid = stable_id(source, context[:200])
            for qa in row["qas"]:
                answers = qa.get("answers") or [d["text"] for d in qa.get("detected_answers", [])]
                if not answers:
                    continue
                spans = [sp for d in qa.get("detected_answers", []) for sp in d.get("char_spans", [])]
                ev = _evidence_units(context, units, spans) if spans else [
                    k for k, u in enumerate(units) if answers[0].lower() in u.lower()]
                items.append(Item(
                    item_id=f"mrqa-{qa['qid']}", domain="mrqa", split=split, group_id=gid,
                    units=units, question=qa["question"], answer=answers[0],
                    answer_type=SPAN_TYPE, evidence=ev, polarity="span",
                    meta={"source": source, "all_answers": answers},
                ))
                if max_items and len(items) >= max_items:
                    return items
    return items


@register("mrqa")
def load_mrqa(in_domain: list[str], out_domain: list[str], max_per_file: int | None = None,
              seed: int = 13) -> list[Item]:
    """``in_domain`` / ``out_domain``: paths to MRQA jsonl(.gz) files."""
    items: list[Item] = []
    for p in in_domain:
        items += _read_file(Path(p), TRAIN, max_per_file)
    group_split(items, seed=seed)
    for p in out_domain:
        items += _read_file(Path(p), TEST_SHIFT, max_per_file)
    return items

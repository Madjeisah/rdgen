"""2010 i2b2/VA assertions (clinical domain, cross-site).

Input: the official ``txt/`` (one sentence per line) and ``ast/`` folders, with
one folder per site (e.g. ``beth/`` and ``partners/``). Each problem concept
becomes a question "Does the patient have <concept>?".

  present  -> yes (pos)
  absent   -> no  (explicit_neg; the negation is in the evidence line)
  possible -> not_stated (uncertain; excluded from RD)
  hypothetical / conditional / associated_with_someone_else -> dropped by default

Shift: train on one site, test on the other.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import CATEGORICAL_TYPE, NO, NOT_STATED, TEST_SHIFT, TRAIN, YES, Item
from ..text import stable_id
from . import group_split, register

_AST = re.compile(r'c="(?P<c>.+?)" (?P<l1>\d+):\d+ (?P<l2>\d+):\d+\|\|t="problem"\|\|a="(?P<a>[a-z_]+)"')
_MAP = {"present": (YES, "pos"), "absent": (NO, "explicit_neg"), "possible": (NOT_STATED, "uncertain")}


def _read_site(site_dir: Path, site: str, split: str, keep: set[str]) -> list[Item]:
    items: list[Item] = []
    for ast in sorted((site_dir / "ast").glob("*.ast")):
        txt = site_dir / "txt" / (ast.stem + ".txt")
        if not txt.exists():
            continue
        lines = [" ".join(l.split()) for l in txt.read_text(encoding="utf-8", errors="replace").splitlines()]
        keep_idx = [k for k, l in enumerate(lines) if l]
        remap = {k: n for n, k in enumerate(keep_idx)}
        units = [lines[k] for k in keep_idx]
        gid = stable_id("i2b2", site, ast.stem)
        seen = set()
        for row in ast.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _AST.search(row)
            if not m or m["a"] not in keep:
                continue
            concept, a = m["c"], m["a"]
            key = (concept.lower(), a)
            if key in seen:
                continue
            seen.add(key)
            line = int(m["l1"]) - 1
            if line not in remap:
                continue
            answer, pol = _MAP[a]
            items.append(Item(
                item_id=f"i2b2-{site}-{ast.stem}-{len(items)}", domain="i2b2_2010", split=split,
                group_id=gid, units=units, question=f"Does the patient have {concept}?",
                answer=answer, answer_type=CATEGORICAL_TYPE, evidence=[remap[line]],
                polarity=pol, meta={"site": site, "concept": concept, "assertion": a},
            ))
    return items


@register("i2b2_2010")
def load_i2b2_2010(root: str, train_site: str = "partners", shift_site: str = "beth",
                   keep_assertions: tuple[str, ...] = ("present", "absent", "possible"),
                   seed: int = 13) -> list[Item]:
    root_p = Path(root)
    keep = set(keep_assertions)
    items = _read_site(root_p / train_site, train_site, TRAIN, keep)
    group_split(items, seed=seed)
    items += _read_site(root_p / shift_site, shift_site, TEST_SHIFT, keep)
    return items

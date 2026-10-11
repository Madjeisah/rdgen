"""BioASQ yes/no questions with supporting snippets.

Snippets are relevant by construction, so each document is the question's
snippets mixed with ``n_distractors`` snippets drawn from other questions.
The question's own snippets are marked as evidence.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..schema import CATEGORICAL_TYPE, NO, TEST_SHIFT, YES, Item
from ..text import stable_id
from . import register


@register("bioasq")
def load_bioasq(path: str, split: str = TEST_SHIFT, n_distractors: int = 6, seed: int = 13,
                max_items: int | None = None) -> list[Item]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    qs = [q for q in data["questions"] if q.get("type") == "yesno" and q.get("snippets")]
    pool = [(q["id"], " ".join(s["text"].split())) for q in qs for s in q["snippets"]]
    rng = random.Random(seed)
    items: list[Item] = []
    for q in qs:
        own = list(dict.fromkeys(" ".join(s["text"].split()) for s in q["snippets"]))
        others = [t for qid, t in pool if qid != q["id"]]
        distract = rng.sample(others, min(n_distractors, len(others)))
        units = own + distract
        order = list(range(len(units)))
        rng.shuffle(order)
        units = [units[k] for k in order]
        evidence = sorted(order.index(k) for k in range(len(own)))
        ans = q["exact_answer"].lower() if isinstance(q["exact_answer"], str) else str(q["exact_answer"]).lower()
        answer = YES if ans.startswith("yes") else NO
        items.append(Item(
            item_id=f"bioasq-{q['id']}", domain="bioasq", split=split,
            group_id=stable_id("bioasq", q["id"]), units=units, question=q["body"],
            answer=answer, answer_type=CATEGORICAL_TYPE, evidence=evidence,
            polarity="pos" if answer == YES else "explicit_neg",
            meta={"n_distractors": len(distract)},
        ))
        if max_items and len(items) >= max_items:
            break
    return items

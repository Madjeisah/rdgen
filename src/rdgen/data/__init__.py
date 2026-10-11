"""Dataset loaders. Each returns ``list[Item]`` in the unified schema.

Raw data is never redistributed by this package: every loader reads files the
user has obtained under the dataset's own licence or data use agreement.
"""
from __future__ import annotations

import random
from typing import Callable

from ..schema import TEST_ID, TRAIN, VALID, Item

LOADERS: dict[str, Callable[..., list[Item]]] = {}


def register(name: str):
    def deco(fn):
        LOADERS[name] = fn
        return fn
    return deco


def load(name: str, **kwargs) -> list[Item]:
    _import_all()
    if name not in LOADERS:
        raise KeyError(f"unknown domain loader {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)


def group_split(items: list[Item], frac=(0.70, 0.15, 0.15), seed: int = 13,
                splits=(TRAIN, VALID, TEST_ID)) -> list[Item]:
    """Assign splits at the group (patient / document) level to prevent leakage.

    Only items whose split is TRAIN are re-assigned; shifted-test items keep
    their split.
    """
    groups = sorted({it.group_id for it in items if it.split == TRAIN})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    cut1 = int(round(frac[0] * n))
    cut2 = cut1 + int(round(frac[1] * n))
    assign = {g: splits[0] for g in groups[:cut1]}
    assign.update({g: splits[1] for g in groups[cut1:cut2]})
    assign.update({g: splits[2] for g in groups[cut2:]})
    for it in items:
        if it.split == TRAIN:
            it.split = assign[it.group_id]
    return items


def _import_all() -> None:
    from . import (  # noqa: F401
        bioasq, contractnli, i2b2_2010, medqaro, mrqa, pubmedqa, qasper, synthetic,
    )

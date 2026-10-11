"""Answerer interface shared by real and mock models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..schema import Item


@dataclass
class Prediction:
    answer: str
    confidence: float                       # max-softmax style confidence in [0, 1]
    label_probs: dict[str, float] = field(default_factory=dict)  # categorical only


class Answerer(Protocol):
    name: str
    family: str

    def predict(self, items: Sequence[Item]) -> list[Prediction]:
        """Answer each item from its own ``units`` (already perturbed if any)."""
        ...

    def logprob(self, items: Sequence[Item], answers: Sequence[str]) -> list[float]:
        """Log-probability of a fixed answer given each item's document.

        Needed by the ContextCite-style baseline, which scores the original
        answer under random context ablations.
        """
        ...

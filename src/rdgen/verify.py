"""Evidence verifiers V(unit, q, a) used by the support closure and the gate.

Two implementations share one interface:

* ``LexicalVerifier`` - dependency-free, used in tests and as a fallback.
* ``NLIVerifier``     - a Hugging Face NLI cross-encoder (multilingual model
  for Romanian). The premise is the unit; the hypothesis states the answer.

Methodology 10.7 requires the verifier used for gating to differ from the one
used for measurement; pass two different instances.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from .normalize import normalize_text
from .schema import NO, NOT_STATED, YES


class Verifier(Protocol):
    name: str

    def support(self, units: Sequence[str], question: str, answer: str) -> list[float]:
        """Support score in [0, 1] for each unit."""
        ...


def hypothesis(question: str, answer: str) -> str:
    q = question.rstrip("?").strip()
    if answer == YES:
        return f"Yes: {q}."
    if answer == NO:
        return f"No: it is not the case that {q[0].lower() + q[1:]}."
    if answer == NOT_STATED:
        return f"The text does not say whether {q[0].lower() + q[1:]}."
    return f"{q}: {answer}."


class LexicalVerifier:
    """Overlap between a unit and the question's content words (+ answer)."""

    name = "lexical"
    _STOP = {"is", "the", "a", "an", "of", "what", "does", "do", "did", "was", "were", "are",
             "patient", "have", "has", "any", "there", "in", "to", "for", "and", "or", "present",
             "state", "following", "contract", "value"}

    def __init__(self, threshold_words: int = 1):
        self.threshold_words = threshold_words

    def _content(self, s: str) -> set[str]:
        return {w for w in normalize_text(s).split() if w not in self._STOP and len(w) > 1}

    def support(self, units: Sequence[str], question: str, answer: str) -> list[float]:
        q = self._content(question)
        a = self._content(answer) if answer not in (YES, NO, NOT_STATED) else set()
        out = []
        for u in units:
            w = self._content(u)
            hit_q = len(q & w)
            hit_a = len(a & w) if a else 0
            if a:
                out.append(1.0 if hit_a and hit_q else (0.5 if hit_a else 0.0))
            else:
                out.append(min(1.0, hit_q / max(1, min(len(q), 2))))
        return out


class NLIVerifier:
    """Entailment probability from an NLI cross-encoder."""

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                 device: str | None = None, batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        from .runtime import log, pick_device
        self.name = f"nli:{model_name}"
        dev = device or pick_device(f"verifier {model_name}")
        log(f"loading verifier {model_name}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = dev
        self.model.to(self.device).eval()
        log(f"verifier on {self.device}")
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_idx = labels.get("entailment", 0)
        self.batch_size = batch_size
        self._torch = torch

    def support(self, units: Sequence[str], question: str, answer: str) -> list[float]:
        hyp = hypothesis(question, answer)
        out: list[float] = []
        for i in range(0, len(units), self.batch_size):
            chunk = list(units[i:i + self.batch_size])
            enc = self.tok(chunk, [hyp] * len(chunk), truncation=True, padding=True,
                           return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                p = self.model(**enc).logits.softmax(-1)[:, self.entail_idx]
            out += p.float().cpu().tolist()
        return out


def multilingual_nli(device: str | None = None) -> NLIVerifier:
    """Default verifier for Romanian (MedQARo)."""
    return NLIVerifier("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device=device)

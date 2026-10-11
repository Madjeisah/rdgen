"""Hugging Face causal-LM answerer (open-weight models only).

* Categorical items: the answer is chosen by scoring each label's
  log-likelihood after the prompt (constrained decoding), giving calibrated
  label probabilities for ATC/DoC and a confidence for every prediction.
* Span items: greedy generation; confidence = exp(mean token log-prob).
* ``evidence_first`` models (trained with the sft_evidence regime) generate
  the evidence list first, then the answer is scored/generated after it.

Requires the ``gpu`` extra (torch, transformers, peft).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..context import ContextConfig, render
from ..prompts import LABEL_WORDS, build_prompt
from ..runtime import Progress, log, pick_device
from ..schema import CATEGORICAL, CATEGORICAL_TYPE, NOT_STATED, Item
from .base import Prediction


@dataclass
class HFConfig:
    model_id: str
    family: str
    adapter: str | None = None          # LoRA adapter dir from training
    lang: str = "en"
    context: ContextConfig = field(default_factory=ContextConfig)
    evidence_first: bool = False
    max_new_tokens: int = 24
    batch_size: int = 8
    dtype: str = "bfloat16"
    device_map: str = "auto"
    chat_template: bool = True


class HFAnswerer:
    def __init__(self, cfg: HFConfig, name: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.name = name or (cfg.model_id + (f"+{cfg.adapter}" if cfg.adapter else ""))
        self.family = cfg.family
        self.torch = torch
        pick_device(f"model {cfg.model_id}")
        log(f"loading model {cfg.model_id}" + (f" with adapter {cfg.adapter}" if cfg.adapter else ""))
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=getattr(torch, cfg.dtype), device_map=cfg.device_map)
        if cfg.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, cfg.adapter)
        self.model = model.eval()
        self.labels = LABEL_WORDS[cfg.lang]
        log(f"model on {self.model.device} ({cfg.dtype})")

    # -- prompt -----------------------------------------------------------
    def _prompt(self, item: Item) -> str:
        user = build_prompt(item, render(item, self.cfg.context), self.cfg.lang, self.cfg.evidence_first)
        if self.cfg.chat_template and getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                add_generation_prompt=True)
        else:
            text = user
        return text if self.cfg.evidence_first else text + "Answer:"

    # -- scoring ------------------------------------------------------------
    def _score_continuations(self, prompts: list[str], conts: list[str]) -> list[float]:
        """Sum log-prob of each continuation given its prompt."""
        torch = self.torch
        out: list[float] = []
        prog = Progress("scoring", len(prompts), min_total=200)
        for i in range(0, len(prompts), self.cfg.batch_size):
            prog.update(i)
            P, C = prompts[i:i + self.cfg.batch_size], conts[i:i + self.cfg.batch_size]
            p_ids = [self.tok(p, add_special_tokens=False).input_ids for p in P]
            c_ids = [self.tok(" " + c, add_special_tokens=False).input_ids for c in C]
            seqs = [a + b for a, b in zip(p_ids, c_ids)]
            L = max(map(len, seqs))
            pad = self.tok.pad_token_id
            ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for k, s in enumerate(seqs):
                ids[k, L - len(s):] = torch.tensor(s)
                att[k, L - len(s):] = 1
            ids, att = ids.to(self.model.device), att.to(self.model.device)
            with torch.no_grad():
                logp = self.model(input_ids=ids, attention_mask=att).logits.log_softmax(-1)
            for k, (a, b) in enumerate(zip(p_ids, c_ids)):
                start = L - len(b)
                tgt = ids[k, start:]
                lp = logp[k, start - 1:L - 1].gather(-1, tgt[:, None]).sum().item()
                out.append(lp)
        prog.update(len(prompts))
        return out

    def _generate(self, prompts: list[str], max_new: int) -> list[tuple[str, float]]:
        torch = self.torch
        res: list[tuple[str, float]] = []
        prog = Progress("generating", len(prompts), min_total=200)
        for i in range(0, len(prompts), self.cfg.batch_size):
            prog.update(i)
            enc = self.tok(prompts[i:i + self.cfg.batch_size], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                        output_scores=True, return_dict_in_generate=True,
                                        pad_token_id=self.tok.pad_token_id)
            new = g.sequences[:, enc.input_ids.shape[1]:]
            scores = torch.stack(g.scores, 1).log_softmax(-1)
            for k in range(new.shape[0]):
                toks = new[k]
                mask = toks != self.tok.pad_token_id
                lp = scores[k].gather(-1, toks[:, None]).squeeze(-1)[mask]
                text = self.tok.decode(toks[mask], skip_special_tokens=True).strip().split("\n")[0]
                conf = math.exp(lp.mean().item()) if lp.numel() else 0.0
                res.append((text, conf))
        prog.update(len(prompts))
        return res

    def _with_evidence(self, prompts: list[str]) -> list[str]:
        gens = self._generate(prompts, 40)
        return [p + g.split("Answer:")[0].rstrip() + "\nAnswer:" for p, (g, _) in zip(prompts, gens)]

    # -- interface -------------------------------------------------------------
    def predict(self, items: Sequence[Item]) -> list[Prediction]:
        prompts = [self._prompt(it) for it in items]
        if self.cfg.evidence_first:
            prompts = self._with_evidence(prompts)
        preds: list[Prediction | None] = [None] * len(items)
        cat = [k for k, it in enumerate(items) if it.answer_type == CATEGORICAL_TYPE]
        if cat:
            P, C, owner = [], [], []
            for k in cat:
                for lab in CATEGORICAL:
                    P.append(prompts[k]); C.append(self.labels[lab]); owner.append((k, lab))
            lps = self._score_continuations(P, C)
            by: dict[int, dict[str, float]] = {}
            for (k, lab), lp in zip(owner, lps):
                by.setdefault(k, {})[lab] = lp
            for k, d in by.items():
                m = max(d.values())
                z = sum(math.exp(v - m) for v in d.values())
                probs = {lab: math.exp(v - m) / z for lab, v in d.items()}
                best = max(probs, key=probs.get)
                preds[k] = Prediction(best, probs[best], probs)
        span = [k for k, it in enumerate(items) if it.answer_type != CATEGORICAL_TYPE]
        if span:
            gens = self._generate([prompts[k] for k in span], self.cfg.max_new_tokens)
            for k, (text, conf) in zip(span, gens):
                if text.lower().strip(" .") in {"not stated", "nk", "unknown"}:
                    text = NOT_STATED
                preds[k] = Prediction(text, conf, {})
        return [p for p in preds]  # type: ignore[misc]

    def logprob(self, items: Sequence[Item], answers: Sequence[str]) -> list[float]:
        prompts = [self._prompt(it) for it in items]
        conts = [self.labels.get(a, a) for a in answers]
        return self._score_continuations(prompts, conts)

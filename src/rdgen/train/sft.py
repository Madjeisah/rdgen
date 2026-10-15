"""LoRA supervised fine-tuning (MedQARo-style attention-only adapters).

Loss is computed on the target tokens only. Hyperparameters default to the
MedQARo optimum (lr 2e-5, rank 8, alpha 16, dropout 0.05, 2 epochs, effective
batch 8) so the Romanian case study is comparable with the published numbers.
Requires the ``gpu`` extra.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from ..context import ContextConfig, render
from ..prompts import build_prompt, target_text
from ..schema import Item


@dataclass
class SFTConfig:
    model_id: str
    out_dir: str
    lang: str = "en"
    evidence_first: bool = False
    context: ContextConfig = field(default_factory=ContextConfig)
    lr: float = 2e-5
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    warmup_steps: int = 100
    max_len: int = 4096
    seed: int = 0
    dtype: str = "bfloat16"


def _encode(tok, item: Item, cfg: SFTConfig, chat: bool):
    user = build_prompt(item, render(item, cfg.context), cfg.lang, cfg.evidence_first)
    if chat and getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True)
    else:
        prompt = user
    tgt = target_text(item, cfg.lang, cfg.evidence_first)
    if not cfg.evidence_first:
        prompt, tgt = prompt + "Answer:", " " + tgt.split("Answer:", 1)[1].strip()
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(tgt, add_special_tokens=False).input_ids + [tok.eos_token_id]
    p = p[-(cfg.max_len - len(t)):]
    return p + t, [-100] * len(p) + t


def train_lora(items: Sequence[Item], cfg: SFTConfig, chat: bool = True) -> str:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    from ..runtime import Progress, log, pick_device
    pick_device(f"training {cfg.model_id}")
    log(f"training LoRA on {cfg.model_id}: {len(items)} items, {cfg.epochs} epochs")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=getattr(torch, cfg.dtype),
                                                 device_map="auto")
    targets = [m for m in cfg.target_modules if any(m in n for n, _ in model.named_modules())]
    if not targets:   # e.g. Phi uses a fused qkv_proj
        targets = ["qkv_proj", "o_proj"]
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                             lora_dropout=cfg.lora_dropout, target_modules=targets,
                                             task_type="CAUSAL_LM"))
    data = [_encode(tok, it, cfg, chat) for it in items]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
    steps = math.ceil(len(data) * cfg.epochs / (cfg.batch_size * cfg.grad_accum))
    sched = get_cosine_schedule_with_warmup(opt, cfg.warmup_steps, max(steps, 1))
    model.train()
    train_log = []
    step = 0
    prog = Progress("training batches", cfg.epochs * math.ceil(len(data) / cfg.batch_size))
    done = 0
    for ep in range(cfg.epochs):
        random.shuffle(data)
        for i in range(0, len(data), cfg.batch_size):
            done += 1
            prog.update(done)
            chunk = data[i:i + cfg.batch_size]
            L = max(len(x) for x, _ in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id)
            lab = torch.full((len(chunk), L), -100)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, (x, y) in enumerate(chunk):
                ids[k, :len(x)] = torch.tensor(x)
                lab[k, :len(y)] = torch.tensor(y)
                att[k, :len(x)] = 1
            out = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device),
                        labels=lab.to(model.device))
            (out.loss / cfg.grad_accum).backward()
            if (i // cfg.batch_size + 1) % cfg.grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    train_log.append({"epoch": ep, "step": step, "loss": out.loss.item()})
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"saving adapter to {out_dir}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    cfg_d = asdict(cfg)
    (out_dir / "train_config.json").write_text(json.dumps(cfg_d, indent=2, default=str))
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))
    return str(out_dir)

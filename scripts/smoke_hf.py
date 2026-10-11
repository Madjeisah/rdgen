"""GPU-path smoke test. Run this first on the machine with torch installed:

    pip install -e ".[gpu,dev]"
    python scripts/smoke_hf.py --model Qwen/Qwen2.5-0.5B-Instruct

It checks, on 40 synthetic items, that the HF answerer returns valid
categorical probabilities and span answers, that log-prob scoring works (for
ContextCite), that the record-dependence probe runs, and that one LoRA epoch
trains and reloads as an adapter. It does not test accuracy.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

try:
    import rdgen  # noqa: F401  (installed with `pip install -e .`)
except ModuleNotFoundError:
    # Not installed: fall back to the source tree next to this script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdgen.context import ContextConfig
from rdgen.data import load
from rdgen.estimators import ProbeConfig, point, probe_model, summarize_probe
from rdgen.models.hf import HFAnswerer, HFConfig
from rdgen.train.sft import SFTConfig, train_lora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()
    items = load("synthetic", n_source=20, n_target=10, seed=0)
    test = [i for i in items if i.split == "test_id"][:40] or items[:40]
    m = HFAnswerer(HFConfig(a.model, "smoke", context=ContextConfig("full"), batch_size=4))
    preds = m.predict(test)
    for p, it in zip(preds, test):
        if it.answer_type == "categorical":
            assert abs(sum(p.label_probs.values()) - 1) < 1e-4, p
        assert 0 <= p.confidence <= 1
    lps = m.logprob(test[:4], [it.answer for it in test[:4]])
    assert all(math.isfinite(x) for x in lps)
    df = probe_model(m, test, ProbeConfig(do_sufficiency=False), filler_items=items)
    print("RD on smoke items:", point(summarize_probe(df, None, n_boot=0), "RD"))
    with tempfile.TemporaryDirectory() as d:
        train = [i for i in items if i.split == "train"][:32]
        train_lora(train, SFTConfig(a.model, d, epochs=1, grad_accum=4, warmup_steps=1))
        m2 = HFAnswerer(HFConfig(a.model, "smoke", adapter=d, context=ContextConfig("full")))
        assert len(m2.predict(test[:4])) == 4
    print("smoke test passed")


if __name__ == "__main__":
    main()

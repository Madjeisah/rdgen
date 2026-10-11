"""Command-line entry point: ``rdgen <command> ...``

  prepare      raw data -> unified JSONL for one domain
  prereg-lock  hash the pre-registration (required before shifted evaluation)
  train        LoRA fine-tune one base model under one regime
  gate         filter an augmentation pool (none | entailment | necessity | both)
  evaluate     one model x one domain -> preds, probes, summary
  aggregate    collect runs into the population table
  analyze      pre-registered analysis, figures, report
  pool         build a controlled augmentation pool for the gate ablation
  ablate       ablation analysis (measurement arms, gate modes / H2, Table 2)
  power        power simulation to size the population
  demo         full pipeline on synthetic domains with mock models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _cmd_prepare(a):
    from .data import load
    from .schema import summarize, write_jsonl
    cfg = yaml.safe_load(Path(a.config).read_text())[a.domain]
    items = load(cfg["loader"], **cfg["args"])
    if "shift_from" in cfg:
        from .schema import TEST_SHIFT
        extra = load(cfg["shift_from"]["loader"], **cfg["shift_from"]["args"])
        for it in extra:
            it.split = TEST_SHIFT
        items += extra
    n = write_jsonl(items, a.out)
    print(json.dumps({"written": n, "summary": summarize(items)}, indent=2))


def _cmd_lock(a):
    from .pipeline.prereg import lock
    print(json.dumps(lock(a.prereg, a.lock), indent=2))


def _load_hf(a, adapter=None):
    from .context import ContextConfig
    from .models.hf import HFAnswerer, HFConfig
    ctx = ContextConfig(a.context, a.max_tokens, a.top_k)
    return HFAnswerer(HFConfig(a.model_id, a.family, adapter=adapter, lang=a.lang, context=ctx,
                               evidence_first=a.regime == "sft_evidence"), name=a.name)


def _cmd_train(a):
    from .context import ContextConfig
    from .schema import read_jsonl
    from .train.regimes import build_training_set
    from .train.sft import SFTConfig, train_lora
    items = read_jsonl(a.data)
    pool = read_jsonl(a.aug_pool) if a.aug_pool else []
    train = build_training_set(items, a.regime, pool, a.cf_rate, a.seed)
    ctx = ContextConfig("retrieve" if a.regime == "retrieval" else "trim", a.max_tokens, a.top_k)
    cfg = SFTConfig(a.model_id, a.out, lang=a.lang, evidence_first=a.regime == "sft_evidence",
                    context=ctx, lr=a.lr, epochs=a.epochs, seed=a.seed)
    print(train_lora(train, cfg))


def _cmd_gate(a):
    from .schema import read_jsonl, write_jsonl
    from .train.gate import apply_gate
    from .verify import LexicalVerifier, NLIVerifier
    pool = read_jsonl(a.pool)
    ver = NLIVerifier(a.verifier) if a.verifier else LexicalVerifier()
    probes = []
    for mid in a.probes:
        from .context import ContextConfig
        from .models.hf import HFAnswerer, HFConfig
        probes.append(HFAnswerer(HFConfig(mid, "probe", lang=a.lang, context=ContextConfig("full"))))
    kept, rep = apply_gate(pool, a.mode, ver, probes, a.tau)
    write_jsonl(kept, a.out)
    print(json.dumps(rep.__dict__, indent=2))


def _cmd_evaluate(a):
    from .estimators import ProbeConfig
    from .pipeline.evaluate import ModelMeta, evaluate_model
    from .schema import read_jsonl
    from .verify import LexicalVerifier, NLIVerifier
    items = read_jsonl(a.data)
    model = _load_hf(a, a.adapter)
    extra = dict(kv.split("=", 1) for kv in a.extra)
    meta = ModelMeta(a.name or a.model_id, a.family, a.size_b, a.regime, a.context, a.seed,
                     a.model_id, extra)
    ver = NLIVerifier(a.verifier) if a.verifier else LexicalVerifier()
    alt = NLIVerifier(a.alt_verifier) if a.alt_verifier else None
    rec = evaluate_model(model, meta, items, a.out, ProbeConfig(closure_mode=a.closure, seed=a.seed),
                         ver, a.prereg, a.lock, a.override_prereg, contextcite=not a.no_contextcite,
                         ablations=a.ablations, alt_verifier=alt,
                         ablation_max_items=a.ablation_max_items)
    print(json.dumps({k: rec[k] for k in ("acc_id", "acc_shift", "RD", "TR") if k in rec}, indent=2))


def _cmd_pool(a):
    from .schema import read_jsonl, write_jsonl
    from .train.pool import prior_contaminated_pool
    pool = prior_contaminated_pool(read_jsonl(a.data), a.n, a.contamination, a.seed)
    print(f"{write_jsonl(pool, a.out)} pool items -> {a.out}")


def _cmd_ablate(a):
    import pandas as pd
    from .analysis.ablation import run_ablation_analysis
    from .analysis.report import load_prereg
    res = run_ablation_analysis(pd.read_csv(a.table), load_prereg(a.prereg), a.out)
    print(json.dumps(res.get("H2", {}), indent=2, default=float))


def _cmd_aggregate(a):
    from .pipeline.aggregate import aggregate
    t = aggregate(a.runs)
    t.to_csv(a.out, index=False)
    print(f"{len(t)} model x domain rows -> {a.out}")


def _cmd_analyze(a):
    import pandas as pd
    from .analysis.report import load_prereg, run_analysis
    res = run_analysis(pd.read_csv(a.table), load_prereg(a.prereg), a.out)
    print(json.dumps(res["verdict"], indent=2))


def _cmd_power(a):
    from .analysis.power import power_curve
    df = power_curve(tuple(a.models), a.families, a.domains, a.beta, a.sims, a.k)
    print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


def _cmd_demo(a):
    from .pipeline.demo import run_demo
    res = run_demo(a.out, n_domains=a.domains)
    print(json.dumps(res["verdict"], indent=2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rdgen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare"); s.add_argument("--domain", required=True)
    s.add_argument("--config", default="configs/domains.yaml"); s.add_argument("--out", required=True)
    s.set_defaults(fn=_cmd_prepare)

    s = sub.add_parser("prereg-lock"); s.add_argument("prereg"); s.add_argument("--lock", default="prereg.lock")
    s.set_defaults(fn=_cmd_lock)

    def model_args(s):
        s.add_argument("--model-id", required=True); s.add_argument("--family", required=True)
        s.add_argument("--name"); s.add_argument("--size-b", type=float, default=float("nan"))
        s.add_argument("--regime", default="sft"); s.add_argument("--lang", default="en")
        s.add_argument("--context", default="trim", choices=["trim", "retrieve", "full"])
        s.add_argument("--max-tokens", type=int, default=2048); s.add_argument("--top-k", type=int, default=10)
        s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("train"); model_args(s)
    s.add_argument("--data", required=True); s.add_argument("--out", required=True)
    s.add_argument("--aug-pool"); s.add_argument("--cf-rate", type=float, default=0.3)
    s.add_argument("--lr", type=float, default=2e-5); s.add_argument("--epochs", type=int, default=2)
    s.set_defaults(fn=_cmd_train)

    s = sub.add_parser("gate"); s.add_argument("--pool", required=True); s.add_argument("--out", required=True)
    s.add_argument("--mode", default="both", choices=["none", "entailment", "necessity", "both"])
    s.add_argument("--probes", nargs="*", default=[]); s.add_argument("--verifier")
    s.add_argument("--tau", type=float, default=0.5); s.add_argument("--lang", default="en")
    s.set_defaults(fn=_cmd_gate)

    s = sub.add_parser("evaluate"); model_args(s)
    s.add_argument("--data", required=True); s.add_argument("--adapter")
    s.add_argument("--out", default="runs"); s.add_argument("--prereg", default="configs/prereg.yaml")
    s.add_argument("--lock", default="prereg.lock"); s.add_argument("--override-prereg", action="store_true")
    s.add_argument("--verifier", help="HF NLI model id for the support closure (default: lexical)")
    s.add_argument("--closure", default="full", choices=["top1", "gold", "full"])
    s.add_argument("--no-contextcite", action="store_true")
    s.add_argument("--ablations", action="store_true", help="score every measurement-ablation arm")
    s.add_argument("--alt-verifier", help="second NLI model id for the verifier_alt arm")
    s.add_argument("--ablation-max-items", type=int)
    s.add_argument("--extra", nargs="*", default=[], help="key=value metadata, e.g. gate_mode=both")
    s.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("pool", help="controlled prior-contaminated augmentation pool")
    s.add_argument("--data", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=2000); s.add_argument("--contamination", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=_cmd_pool)

    s = sub.add_parser("ablate", help="ablation analysis: measurement arms + gate (H2)")
    s.add_argument("--table", default="population.csv")
    s.add_argument("--prereg", default="configs/prereg.yaml"); s.add_argument("--out", default="report/ablation")
    s.set_defaults(fn=_cmd_ablate)

    s = sub.add_parser("aggregate"); s.add_argument("--runs", default="runs")
    s.add_argument("--out", default="population.csv"); s.set_defaults(fn=_cmd_aggregate)

    s = sub.add_parser("analyze"); s.add_argument("--table", default="population.csv")
    s.add_argument("--prereg", default="configs/prereg.yaml"); s.add_argument("--out", default="report")
    s.set_defaults(fn=_cmd_analyze)

    s = sub.add_parser("power"); s.add_argument("--models", type=int, nargs="+", default=[12, 24, 36, 48])
    s.add_argument("--families", type=int, default=6); s.add_argument("--domains", type=int, default=4)
    s.add_argument("--beta", type=float, default=0.5); s.add_argument("--sims", type=int, default=200)
    s.add_argument("--k", type=int, default=3); s.add_argument("--out")
    s.set_defaults(fn=_cmd_power)

    s = sub.add_parser("demo"); s.add_argument("--out", default="demo_out")
    s.add_argument("--domains", type=int, default=4); s.set_defaults(fn=_cmd_demo)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

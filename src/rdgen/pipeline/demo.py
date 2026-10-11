"""End-to-end demo on synthetic domains with mock models.

Runs the full protocol - population, per-domain evaluation with probes,
measurement-ablation arms and baselines, the gate ablation with real gate
filtering, aggregation, the pre-registered analysis and the ablation analysis
- on a CPU. It validates the code path, not the hypotheses: mock models are
built so that reading transfers, and how training on a gated pool changes a
mock model's reliance is an assumption encoded below (``_gated_reliance``).

Mock families differ in reading noise and in sensitivity to input disruption,
so the deletion control and its correction model have something to remove.
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml

from ..analysis.ablation import run_ablation_analysis
from ..analysis.report import run_analysis
from ..data.synthetic import default_specs, load_synthetic, make_record
from ..estimators import ProbeConfig
from ..models.mock import MockAnswerer
from ..perturb import delete, support_closure
from ..schema import TRAIN
from ..train.gate import apply_gate
from ..train.pool import prior_contaminated_pool
from ..verify import LexicalVerifier
from .aggregate import aggregate
from .evaluate import ModelMeta, evaluate_model
from .prereg import lock

REGIME_RELIANCE = {"zeroshot": 0.25, "sft": 0.55, "sft_evidence": 0.65, "abstain_aug": 0.75,
                   "retrieval": 0.6}
GATE_MODES = ("none", "entailment", "necessity", "both")


def _domains(n_domains: int, seed: int) -> dict[str, tuple[list, object]]:
    out = {}
    for d in range(n_domains):
        src, tgt = default_specs(k=6, m=2, flip_fraction=0.3 + 0.15 * d)
        items = load_synthetic(n_source=240, n_target=120, seed=seed + d, specs=(src, tgt),
                               domain=f"synth{d}")
        out[f"synth{d}"] = (items, tgt)
    return out


def _gated_pools(items, tgt_spec, domain: str, seed: int) -> dict[str, tuple[list, float]]:
    """Augmentation pool = generated target-style records (true labels) plus
    prior-consistent contamination; filtered by each gate mode. Returns the
    kept pool and the share of kept items guessable from source priors."""
    rng = random.Random(seed)
    gen = []
    for r in range(80):
        gen += make_record(f"gen{r}", tgt_spec, TRAIN, rng, domain)
    pool = gen + prior_contaminated_pool(items, n=200, contamination=0.6, seed=seed)
    probe = MockAnswerer("gate-probe", family="probe", reliance=0.0).fit_prior(items)
    out = {}
    for mode in GATE_MODES:
        kept, _ = apply_gate(pool, mode, LexicalVerifier(threshold_words=2), [probe])
        if kept:
            ablated = [delete(it, support_closure(it, "gold"), "chk") for it in kept]
            guess = sum(p.answer == it.answer for p, it in zip(probe.predict(ablated), kept)) / len(kept)
        else:
            guess = 0.0
        out[mode] = (kept, guess)
    return out


def _gated_reliance(sft_reliance: float, guessable_share: float) -> float:
    """Assumption for the mock: training on items that cannot be answered from
    priors teaches reading; training on prior-guessable items teaches guessing."""
    return min(0.97, max(0.05, sft_reliance + 0.25 * (1 - guessable_share) - 0.25 * guessable_share))


def run_demo(out_dir: str | Path = "demo_out", n_domains: int = 4, n_families: int = 5,
             sizes_per_family: int = 2, regimes=("zeroshot", "sft", "abstain_aug"),
             seeds=(0,), seed: int = 0, prereg_path: str | None = None,
             ablations: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prereg_path = prereg_path or str(Path(__file__).resolve().parents[3] / "configs" / "prereg.yaml")
    lock_path = out / "prereg.lock"
    lock(prereg_path, lock_path)
    rng = random.Random(seed)
    verifier, alt_verifier = LexicalVerifier(), LexicalVerifier(threshold_words=2)
    fam = {f"fam{f}": (rng.uniform(0.0, 0.08), rng.uniform(0.0, 4.0)) for f in range(n_families)}

    def run(name, family, reliance, noise, disruption, abstain, items, prior_items, meta_kw):
        m = MockAnswerer(name, family=family, reliance=reliance, read_noise=noise,
                         disruption=disruption, abstain_when_silent=abstain).fit_prior(prior_items)
        meta = ModelMeta(name=name, family=family, context="trim", **meta_kw)
        evaluate_model(m, meta, items, out / "runs", ProbeConfig(seed=meta_kw["seed"]), verifier,
                       prereg_path=prereg_path, lock_path=str(lock_path), contextcite=True,
                       n_boot=50, ablations=ablations, alt_verifier=alt_verifier)

    for dname, (items, tgt) in _domains(n_domains, seed).items():
        pools = _gated_pools(items, tgt, dname, seed)
        for fname, (noise, disruption) in fam.items():
            for s in range(sizes_per_family):
                base = f"{fname}-size{s}"
                for sd in seeds:
                    rel = {r: min(0.97, max(0.05, REGIME_RELIANCE[r] + 0.08 * s + rng.gauss(0, 0.06)))
                           for r in regimes}
                    common = dict(size_b=1.0 + 4 * s, seed=sd, base_model=base)
                    for reg in regimes:
                        run(f"{base}-{reg}-s{sd}", fname, rel[reg], noise, disruption,
                            reg == "abstain_aug", items, items,
                            dict(regime=reg, extra={"true_reliance": rel[reg]}, **common))
                    for mode, (kept, guess) in pools.items():
                        r_g = _gated_reliance(rel["sft"], guess)
                        run(f"{base}-gated-{mode}-s{sd}", fname, r_g, noise, disruption, False,
                            items, list(items) + kept,
                            dict(regime="gated", extra={"true_reliance": r_g, "gate_mode": mode,
                                                        "pool_guessable": guess}, **common))
    table = aggregate(out / "runs")
    table.to_csv(out / "population.csv", index=False)
    prereg = yaml.safe_load(Path(prereg_path).read_text())
    prereg["analysis"]["n_boot_family"] = 50
    res = run_analysis(table, prereg, out / "report")
    res["ablation"] = run_ablation_analysis(table, prereg, out / "report" / "ablation")
    return res

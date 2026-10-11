"""Emit one shell command per (domain x model x regime x seed) job.

    python scripts/make_jobs.py --domains mrqa qasper --pilot > jobs.txt

Ablations:
    --ablations                      every evaluation also scores all measurement arms
    --gate-ablation --stage gate     pool-filtering jobs, one per domain x gate mode (run first)
    --gate-ablation                  gated models for every gate mode (H2)

    # then run with GNU parallel, a SLURM array, or your scheduler of choice:
    #   sbatch --array=1-$(wc -l < jobs.txt) scripts/slurm_array.sh jobs.txt

Training jobs must finish before their evaluation jobs; each line chains the
two with && so one line is one self-contained job.
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="configs/population.yaml")
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--verifier", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    ap.add_argument("--pilot", action="store_true", help="use the pilot subset from the config")
    ap.add_argument("--ablations", action="store_true",
                    help="score every measurement-ablation arm in each evaluation")
    ap.add_argument("--alt-verifier", default="cross-encoder/nli-deberta-v3-base",
                    help="second verifier for the verifier_alt arm (must differ from --verifier)")
    ap.add_argument("--ablation-max-items", type=int, default=1500)
    ap.add_argument("--no-contextcite", action="store_true",
                    help="skip the ContextCite baseline (the slowest step); implied by --pilot")
    ap.add_argument("--gate-ablation", action="store_true",
                    help="emit the gate-mode arms (none/entailment/necessity/both) for H2")
    ap.add_argument("--gate-probes", nargs="+",
                    default=["HuggingFaceTB/SmolLM2-1.7B-Instruct", "ibm-granite/granite-3.1-2b-instruct"],
                    help="probe models for the necessity gate; must come from families outside the "
                         "measured population (Goodhart guard)")
    ap.add_argument("--gate-verifier", default="facebook/bart-large-mnli",
                    help="verifier for the entailment gate; must differ from both measurement verifiers")
    ap.add_argument("--stage", choices=["models", "gate"], default="models",
                    help="'gate' emits only the pool-filtering jobs (run first with --gate-ablation)")
    a = ap.parse_args()

    pop = yaml.safe_load(Path(a.population).read_text())
    fams, regimes, seeds = pop["families"], pop["regimes"], pop["seeds"]
    if a.pilot:
        fams = {k: v for k, v in fams.items() if k in pop["pilot"]["families"]}
        regimes, seeds = pop["pilot"]["regimes"], pop["pilot"]["seeds"]
    gate_modes = pop.get("gate_ablation", {}).get("modes", ["none", "entailment", "necessity", "both"])
    if a.gate_ablation:
        regimes = [r for r in regimes if r != "gated"]

    for dom in a.domains:
        data = f"{a.data_dir}/{dom}.jsonl"
        lang = "ro" if dom == "medqaro" else "en"
        verifier = ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
                    if lang == "ro" else a.verifier)
        abl = (f" --ablations --alt-verifier {a.alt_verifier} --ablation-max-items {a.ablation_max_items}"
               if a.ablations else "")
        if a.stage == "gate":
            # pool filtering per gate mode: run this stage before the gated jobs
            gate_verifier = "joeddav/xlm-roberta-large-xnli" if lang == "ro" else a.gate_verifier
            measured = {m["id"] for ms in pop["families"].values() for m in ms}
            clash = measured & set(a.gate_probes)
            if clash:
                raise SystemExit(f"gate probes {sorted(clash)} are in the measured population")
            for mode in gate_modes:
                print(f"rdgen gate --pool data/aug/{dom}.pool.jsonl --mode {mode} "
                      f"--verifier {gate_verifier} --probes {' '.join(a.gate_probes)} --lang {lang} "
                      f"--out data/aug/{dom}.{mode}.jsonl")
            continue
        for fam, models in fams.items():
            for m in models:
                mid, size = m["id"], m["size_b"]
                short = mid.split("/")[-1]
                arms = [(reg, None) for reg in regimes]
                if a.gate_ablation:
                    arms += [("gated", mode) for mode in gate_modes]
                for reg, mode in arms:
                    for sd in seeds:
                        tag = f"{reg}-{mode}" if mode else reg
                        name = f"{short}-{tag}-s{sd}"
                        ctx = "retrieve" if reg == "retrieval" else "trim"
                        common = (f"--model-id {shlex.quote(mid)} --family {fam} --size-b {size} "
                                  f"--regime {reg} --seed {sd} --lang {lang} --context {ctx}")
                        extra = f" --extra gate_mode={mode}" if mode else ""
                        ev = (f"rdgen evaluate {common} --name {name} --data {data} "
                              f"--out {a.runs} --verifier {verifier}{abl}{extra}"
                              f"{' --no-contextcite' if (a.no_contextcite or a.pilot) else ''}")
                        if reg == "zeroshot":
                            print(ev)
                            continue
                        adapter = f"{a.adapters}/{dom}/{name}"
                        pool = f"data/aug/{dom}.{mode or 'both'}.jsonl"
                        aug = f" --aug-pool {pool}" if reg == "gated" else ""
                        tr = f"rdgen train {common} --data {data} --out {adapter}{aug}"
                        print(f"{tr} && {ev} --adapter {adapter}")


if __name__ == "__main__":
    main()

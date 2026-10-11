# rdgen — record-dependence generalisation

Code for testing whether a model's dependence on the document, measured
in-domain and without target labels, predicts its accuracy under distribution
shift in document-grounded QA, beyond what in-domain accuracy predicts.

The same model population runs in four domains (clinical, legal, scientific,
general) under one pre-registered analysis.

## Protocol → code

| Protocol step | Where |
| --- | --- |
| Unified task format (document units, question, answer, evidence, polarity) | `src/rdgen/schema.py` |
| Loaders: MRQA, Qasper, PubMedQA, BioASQ, ContractNLI, 2010 i2b2/VA, MedQARo, synthetic | `src/rdgen/data/` |
| Answer normalisation and correctness | `src/rdgen/normalize.py` |
| Support closure, evidence and control deletion, sufficiency, substitution | `src/rdgen/perturb.py` |
| RD, RD_raw, SUF, TR, UC, AB with cluster bootstrap | `src/rdgen/estimators.py` |
| 1. Model population (open-weight HF models, mock models) | `configs/population.yaml`, `src/rdgen/models/` |
| Regimes: zeroshot, sft, sft_evidence, abstain_aug, gated, retrieval | `src/rdgen/train/regimes.py`, `train/sft.py` |
| Necessity / entailment gate | `src/rdgen/train/gate.py` |
| 2. Per-domain evaluation (ID accuracy, ID RD, shifted accuracy) | `src/rdgen/pipeline/evaluate.py` |
| 3. Hierarchical fits, random-effects pooling, success rule, forest plot | `src/rdgen/analysis/fit.py`, `analysis/figures.py` |
| 4. Interventional test (accuracy-matched pairs) | `src/rdgen/analysis/selection.py` |
| 5. Baselines: ID accuracy, agreement-on-the-line, ATC, DoC, ContextCite-style | `src/rdgen/baselines/` |
| 6. Ablations: measurement arms and gate modes (H2) | `src/rdgen/ablation.py`, `analysis/ablation.py`, `train/pool.py`, `rdgen ablate` |
| 7. Power analysis to size the population | `src/rdgen/analysis/power.py`, `rdgen power` |
| Pre-registration lock on the shifted split | `src/rdgen/pipeline/prereg.py`, `configs/prereg.yaml` |

## Install

```bash
pip install -e ".[dev]"          # analysis, estimators, tests (CPU)
pip install -e ".[gpu,dev]"      # + torch, transformers, peft for real models
```

## Quick check (CPU, about 5 minutes)

```bash
pytest -q                        # 51 tests
rdgen demo --out demo_out        # whole pipeline + ablations on synthetic domains (about 8 min)
```

The demo builds four synthetic domains whose priors and documentation
conventions shift between source and target, a population of mock models that
mix reading and guessing in known proportions, and runs evaluation, baselines,
aggregation, the pre-registered analysis and all figures. It validates the
code path, not the hypothesis: the mock models are built so that reading
transfers.

## Running the real study

```bash
# 0. GPU smoke test (run first; checks the HF answerer, scoring, probes, LoRA)
python scripts/smoke_hf.py --model Qwen/Qwen2.5-0.5B-Instruct

# 1. Data: obtain raw files under each licence / DUA, set paths in configs/domains.yaml
#    (MRQA is public: bash scripts/download_mrqa.sh fetches it to the configured paths)
rdgen prepare --domain mrqa --out data/processed/mrqa.jsonl

# 2. Lock the pre-registration (shifted-split scoring is refused without it)
rdgen prereg-lock configs/prereg.yaml

# 3. Pilot on MRQA, then size the population
python scripts/make_jobs.py --domains mrqa --pilot > pilot_jobs.txt
sbatch --array=1-$(wc -l < pilot_jobs.txt) scripts/slurm_array.sh pilot_jobs.txt
#   no Slurm (single workstation): bash scripts/run_local.sh pilot_jobs.txt [--gpus 0,1]
rdgen aggregate --runs runs --out pilot.csv
rdgen power --models 12 24 36 48 --beta <pilot estimate> --families 6 --domains 4

# 4. Full population (1,350 jobs for 5 domains with the default config)
python scripts/make_jobs.py --domains mrqa qasper contractnli i2b2_2010 medqaro > jobs.txt

# 5. Analysis
rdgen aggregate --runs runs --out population.csv
rdgen analyze --table population.csv --prereg configs/prereg.yaml --out report
```

`report/` contains `report.md` (states whether the pre-registered success rule
is met), per-domain fits, regret, interventional results, and figures 2–4.

## Ablations

Two kinds, both reported by `rdgen ablate` (Table 2, Table 3, an appendix
figure, and LaTeX tables ready for the paper).

**Measurement ablations** re-score the same models with different estimator
choices, so they need no retraining. `rdgen evaluate --ablations` scores every
arm in one pass, sharing predictions through a cache (`src/rdgen/ablation.py`):

| Arm | Question it answers |
| --- | --- |
| primary | pre-registered RD: full closure, union control correction, deletion on pos + span |
| no_control | is apparent dependence just input disruption? |
| control_difference, control_xor | does the control-correction model matter? |
| closure_top1, closure_gold | does handling repeated evidence, and the verifier, matter? |
| del_explicit_neg, del_all_pol | why handle polarity separately? |
| substitution, sufficiency | independent estimators |
| verifier_alt | sensitivity to the verifier |
| contextcite | is gold evidence needed at all? |

Each arm is judged on validity (agreement with ground-truth reliance; only
available on synthetic data), agreement with the primary arm, pooled effect
per SD, domains positive, gain in explained variance, out-of-sample gain and
selection regret. Robustness subgroups re-fit the primary estimator on trained
models only, on models never trained with a gate (Goodhart check), and by
context strategy.

**Gate ablation (H2)** trains one gated model per gate mode
(none / entailment / necessity / both) and compares each with the sft model of
the same base model, seed and domain. H2 is tested on a difference,
D = gain(mode) − 0.7 × gain(none), not on a ratio, because the retained share
is unstable when the ungated gain is near zero.

```bash
rdgen pool --data data/processed/mrqa.jsonl --out data/aug/mrqa.pool.jsonl   # or your generator's pool
python scripts/make_jobs.py --domains mrqa --gate-ablation --stage gate > gate_jobs.txt   # run first
python scripts/make_jobs.py --domains mrqa --gate-ablation --ablations > jobs.txt
rdgen aggregate --runs runs --out population.csv
rdgen ablate --table population.csv --out report/ablation
```

Gate probes and the gate verifier default to models outside the measured
population, and `make_jobs.py` refuses probes that are in it.

What the synthetic demo shows (code-path check with mock models; the
directions are the point, not the values):

| Arm | Validity | Lesson |
| --- | --- | --- |
| primary | 0.91 | |
| no_control | 0.85 | disruption makes guessers look like readers |
| control_difference | 0.51 | subtraction over-corrects and compresses the scale |
| control_xor | 0.14 | wrong model of disruption |
| closure_top1 | 0.89 | underestimates when facts repeat |
| del_explicit_neg | 0.47 | the polarity problem: halves validity, removes the effect |
| substitution | 0.98 | best independent cross-check |

Predictive power alone does not separate several arms; validity does. Real
data has no ground-truth reliance, so the synthetic validity study belongs in
the paper's appendix, and agreement with the substitution estimator is the
validity proxy on real domains.

## Design decisions worth knowing

**Control correction model.** Evidence deletion and input disruption both
change answers, but they do not add: disruption can hit an item whose answer
already changed. The primary estimator treats them as independent causes,
RD = (p − c) / (1 − c), where p is the evidence-deletion change rate and c the
control change rate. Plain subtraction is kept as an ablation arm.

**Explicit negatives are measured by substitution, not deletion.** A model
that reads with a closed-world convention keeps answering "no" when an
explicit negation is deleted, so deletion-based RD on those items is about 0
even for a model that always reads (`tests/test_estimators.py::test_explicit_negatives_are_invisible_to_deletion`).
Deletion therefore applies to positive and span items only; explicit
negatives enter through the substitution estimator TR. Implicit negatives and
uncertain labels are excluded from RD and used for abstention analysis.

**RD is conditional on correct answers.** It measures the share of correct
answers that depend on the record, so it can exceed a model's raw reading rate:
guessing produces more wrong answers than reading does.

**RD adds to in-domain accuracy; it does not replace it.** On its own it is a
poor model selector. The claim, the fits and the `combined` criterion all test
whether it adds information beyond in-domain accuracy.

**Goodhart guard.** The verifier and probe models used by the gate must differ
from those used to measure RD; pass different instances. Results for models
never trained with a gate are reported separately in the analysis.

**Shifted split discipline.** `rdgen evaluate` refuses to score `test_shift`
unless `prereg.lock` matches the current `configs/prereg.yaml`. Overrides are
possible (`--override-prereg`) but recorded in every run's provenance.

## Status and known gaps

- Tested here: schema, loaders (on format fixtures), perturbations,
  estimators, baselines, gate, analysis, power simulation, CLI, and the full
  synthetic pipeline.
- **Not run here:** the Hugging Face answerer, LoRA training and NLI
  verifier. They compile, but this environment cannot download PyTorch or
  models. Run `scripts/smoke_hf.py` before anything else on GPU.
- ContractNLI: whether the released metadata supports an EDGAR vs web split
  must be checked; `default_source_of` is a heuristic. Fallback:
  `shift: hypothesis`.
- MedQARo: the loader reads a CSV/JSONL export with configurable columns.
  "Nu" answers need the WP1 polarity annotation; without it they are marked
  `unknown_neg` and excluded from RD.
- Model ids in `configs/population.yaml` are a starting point; check
  availability, licences and gating on the Hub.
- Power-simulation defaults are placeholders; replace them with pilot
  estimates.

## Data governance

No raw data is redistributed. MIMIC-derived data requires PhysioNet
credentials and must not be sent to third-party APIs; n2c2/i2b2 data requires
the n2c2 data use agreement; MedQARo is CC BY-NC-SA 4.0, so derived
annotations must be released under the same licence.

## Licence

Code: MIT. Datasets keep their own licences.

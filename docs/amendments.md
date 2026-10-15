# Pre-registration amendments

Changes made after `configs/prereg.yaml` was locked. Each entry states what
changed, why, what evidence prompted it, and whether target-split results
informed the decision.

## 2026-10-15 — Substitution estimator: type-matched substitutes for span answers

**What changed.** For span answers without a question template (MRQA, Qasper
extractive answers), the substitute value v' was drawn at random from all
training answers. It is now drawn from answers of the same coarse type (year,
date, number, name, phrase), preferring similar length, from the same source
dataset, and never from strings already in the document. Items with no
same-type substitute are skipped rather than given a mismatched one. The probe
also records the model's answer to each substituted document and classifies
non-followed cases (kept original answer / abstained / partly followed /
other), reported as `TR_kept_original`, `TR_abstained`, `TR_followed_partly`,
`TR_other`.

**Why.** A random substitute of a different type (e.g. a year replaced by a
noun phrase) makes the edited document incoherent, so a model that declines to
repeat it is not thereby shown to be guessing. TR then measured plausibility of
the edit rather than reading.

**Evidence.** Pilot (MRQA), two zero-shot models: RD 0.86 / 0.95 against TR
0.34 / 0.28, on in-domain items only. The diagnosis and the fix use in-domain
data only; no target-split result was used.

**Unchanged.** The primary estimator (RD, union control, pos + span), the
hypotheses, success criteria and analysis.

#!/usr/bin/env bash
# Download MRQA 2019 data into the paths configs/domains.yaml expects.
# Run from the repository root:  bash scripts/download_mrqa.sh
# BioASQ is not included: it is distributed by BioASQ and needs registration.
set -euo pipefail
BASE=https://s3.us-east-2.amazonaws.com/mrqa/release/v2
mkdir -p data/raw/mrqa/train data/raw/mrqa/dev
for f in SQuAD NewsQA TriviaQA-web SearchQA HotpotQA NaturalQuestionsShort; do
  wget -c -O "data/raw/mrqa/train/$f.jsonl.gz" "$BASE/train/$f.jsonl.gz"
done
for f in DROP DuoRC.ParaphraseRC RACE RelationExtraction TextbookQA; do
  wget -c -O "data/raw/mrqa/dev/$f.jsonl.gz" "$BASE/dev/$f.jsonl.gz"
done
echo "Done. Next: rdgen prepare --domain mrqa --out data/processed/mrqa.jsonl"

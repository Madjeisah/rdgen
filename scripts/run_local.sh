#!/usr/bin/env bash
# Run a jobs file (one shell command per line, from scripts/make_jobs.py) on
# this machine, without Slurm.
#
#   bash scripts/run_local.sh pilot_jobs.txt            # all GPUs visible, one job at a time
#   bash scripts/run_local.sh pilot_jobs.txt --gpus 0,1 # one worker per listed GPU
#
# Each job writes logs/<jobs-name>/<line>.out. A job that finishes is marked
# done, so re-running the same command resumes and skips it. Failed jobs are
# listed at the end and in logs/<jobs-name>/failed.txt; the run carries on.
set -uo pipefail

JOBS=${1:?usage: run_local.sh JOBS_FILE [--gpus 0,1,...]}
shift
GPUS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus) GPUS=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

NAME=$(basename "$JOBS" .txt)
LOG=logs/$NAME
mkdir -p "$LOG/done"
: > "$LOG/failed.txt"
N=$(grep -c '' "$JOBS")

worker() {  # worker <index> <count> <gpu or empty>
  local w=$1 k=$2 gpu=$3 i=0 cmd
  while IFS= read -r cmd || [[ -n $cmd ]]; do
    i=$((i + 1))
    (( (i - 1) % k == w )) || continue
    [[ -z ${cmd// } || $cmd == \#* ]] && continue
    [[ -e $LOG/done/$i ]] && { echo "[$i/$N] already done, skipping"; continue; }
    echo "[$i/$N]${gpu:+ gpu $gpu} start"
    if ( [[ -n $gpu ]] && export CUDA_VISIBLE_DEVICES=$gpu
         echo "[$(date -u +%FT%TZ)] $cmd"; eval "$cmd" ) > "$LOG/$i.out" 2>&1; then
      touch "$LOG/done/$i"; echo "[$i/$N] done"
    else
      echo "$i: $cmd" >> "$LOG/failed.txt"; echo "[$i/$N] FAILED (see $LOG/$i.out)"
    fi
  done < "$JOBS"
}

if [[ -z $GPUS ]]; then
  worker 0 1 ""
else
  IFS=, read -ra G <<< "$GPUS"
  for w in "${!G[@]}"; do worker "$w" "${#G[@]}" "${G[$w]}" & done
  wait
fi

if [[ -s $LOG/failed.txt ]]; then
  echo "$(wc -l < "$LOG/failed.txt") job(s) failed; see $LOG/failed.txt. Re-run the same command to retry them."
  exit 1
fi
echo "All $N jobs done. Logs in $LOG/."

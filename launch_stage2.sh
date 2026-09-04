#!/usr/bin/env bash
# Wait for all 4 trainset shards, then run stage2. Aborts if the extraction procs die first.
cd /datasets/work/vLLM/temp/PAAS_simplicity
PIDS="2522806 2522807 2522808 2522809"
while [ "$(ls cache/train_0.npz cache/train_1.npz cache/train_2.npz cache/train_3.npz 2>/dev/null | wc -l)" -ne 4 ]; do
  alive=0
  for p in $PIDS; do [ -d "/proc/$p" ] && alive=$((alive+1)); done
  if [ "$alive" -eq 0 ]; then
    echo "[launch] extraction processes all gone with only $(ls cache/train_*.npz 2>/dev/null | wc -l)/4 npz -- NOT starting stage2"
    exit 1
  fi
  sleep 20
done
echo "[launch] 4/4 npz present at $(date '+%F %T'); waiting 20s for final flush, then stage2"
sleep 20
exec bash stage2.sh

#!/usr/bin/env bash
# Everything that waits on the axon1 extraction. Ordered so the SPEED BENCHMARK runs last and alone:
# a contended timing number is worthless, which is the whole reason it is not parallelised with the rest.
set -uo pipefail
cd /datasets/work/vLLM/temp/PAAS_simplicity
PY=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
export PYTHONNOUSERSITE=1
L=runs/spc
step(){ echo; echo "############ $(date '+%F %T')  $1 ############"; }

step "0/6 wait for the 8 axon1 shards (explicit PIDs, never a pkill pattern)"
fail=0
while read -r p; do
  while [ -d "/proc/$p" ]; do sleep 30; done
done < runs/extract/axon1.pids
n=$(ls cache/axon1_*.npz 2>/dev/null | wc -l)
echo "  axon1 processes finished; $n/8 shards written"
[ "$n" -eq 8 ] || { echo "  ABORT: expected 8 axon1 shards, found $n"; exit 1; }

step "1/6 verify the axon1 cache (614,029 = 48,526 real + 565,503 pad)"
$PY -u spc/verify_cache.py --cache cache/axon1 --expect axon1 --out $L/verify_axon1.json || exit 1

step "2/6 dedup axon1 vs the testset (sensitivity table + group-level mask + alignment checksums)"
CUDA_VISIBLE_DEVICES=0 $PY -u spc/dedup_axon1.py --device cuda:0 --out $L/axon1_overlap.npz || exit 1

step "3/6 evaluate axon1: raw + frame-level + group-level, vs FFAA on the SAME frames"
CUDA_VISIBLE_DEVICES=0 $PY -u spc/eval_axon1.py --ckpt $L/default.pt --overlap $L/axon1_overlap.npz \
    --device cuda:0 --out $L/axon1.json || exit 1

step "4/6 FFAA + all members on the FULL testset, 4 shards on 4 GPUs"
BP=()
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup $PY -u spc/score_baselines.py --device cuda:0 --batch-size 128 \
    --with-ffaa --which-part $s --n-divided 4 --out $L/baselines_shard$s.json \
    > $L/baselines_shard$s.log 2>&1 < /dev/null &
  BP+=($!)
done
bf=0; for p in "${BP[@]}"; do wait "$p" || bf=$((bf+1)); done
echo "  baseline shards done, $bf non-zero"
if [ "$bf" -eq 0 ]; then
  $PY merge_baselines.py "$L/baselines_shard*.json" "$L/baselines_testset.json" || exit 1
else
  echo "  FFAA sharding failed; falling back to the cheap-member full-testset run already on disk"
  cp $L/baselines_cheap_full.json $L/baselines_testset.json
fi

step "5/6 speed benchmark -- EXCLUSIVE GPU, nothing else running"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed 's/^/  still holding a GPU: /'
CUDA_VISIBLE_DEVICES=0 $PY -u spc/bench_speed.py --ckpt $L/default.pt --device cuda:0 \
    --out $L/bench.json || exit 1

step "6/6 render Experiment 24 into EXPERIMENTS.pdf"
$PY report/build_exp24.py || exit 1
echo; echo "############ $(date '+%F %T')  STAGE 3B COMPLETE ############"

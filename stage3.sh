#!/usr/bin/env bash
# Stage 3: the comparison work. Runs after stage2.sh has produced runs/spc/default.pt.
#
# Ordering rationale: the axon1 feature extraction is the long pole (614,029 VIDEO FRAMES, decoded
# not read), so it starts first and the cheap baseline scoring runs alongside it on a different GPU.
set -uo pipefail
cd /datasets/work/vLLM/temp/PAAS_simplicity
PY=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
export PYTHONNOUSERSITE=1
L=runs/spc
step() { echo; echo "############ $(date '+%F %T')  $1 ############"; }

[ -f $L/default.pt ] || { echo "no $L/default.pt -- run stage2.sh first"; exit 2; }

step "1/5 axon1 feature extraction: 8 shards over GPUs 1,2,3 (2-3 per GPU) -- BACKGROUND"
# 2 procs per GPU so single-threaded cv2 decode overlaps GPU encode. GPU 0 is left for the baselines.
AX_PIDS=()
for s in 0 1 2 3 4 5 6 7; do
  gpu=$(( 1 + (s % 3) ))
  CUDA_VISIBLE_DEVICES=$gpu AXON1_THREADS=3 AXON1_CV_THREADS=2 nohup $PY -u spc/extract_axon1.py \
     --out cache/axon1 --which-part $s --n-divided 8 --batch-size 64 \
     > runs/extract/axon1_$s.log 2>&1 < /dev/null &
  AX_PIDS+=($!)
  echo "  shard $s -> GPU $gpu (pid ${AX_PIDS[-1]})"
done
printf '%s\n' "${AX_PIDS[@]}" > runs/extract/axon1.pids

step "2/5 baseline models on the SAME testset, GPU 0 (mids9c / GSD / SeLop / FFAA)"
# FFAA included: the recorded number has no per-class pad/deepfake recall, and putting the strongest
# baseline through the SAME harness as everything else is what makes the comparison table a
# measurement instead of a collage. ~9.3 fps -> ~55 min, which overlaps the axon1 decode above.
$PY -u spc/score_baselines.py --device cuda:0 --batch-size 64 --with-ffaa \
    --out $L/baselines_testset.json || {
  echo "  FFAA run failed; retrying WITHOUT ffaa so the cheap baselines are still measured"
  $PY -u spc/score_baselines.py --device cuda:0 --batch-size 64 --out $L/baselines_testset.json || exit 1
}

step "3/5 wait for axon1 extraction"
# Wait on EXPLICIT PIDs, never `pgrep -f`/`pkill -f` on a pattern: a pattern match also matches the
# command line of the very shell doing the matching. That mistake killed a launcher mid-loop earlier
# in this project and produced two false "still running" reports.
fail=0
for p in "${AX_PIDS[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "  axon1 extraction done, $fail shard(s) exited non-zero"
# Counting failures and then continuing is worse than not counting them: the next step would verify a
# PARTIAL cache and, with --expect none, happily pass it.
[ "$fail" -eq 0 ] || { echo "  ABORT: $fail axon1 shard(s) failed -- see runs/extract/axon1_*.log"; exit 1; }
n_npz=$(ls cache/axon1_*.npz 2>/dev/null | wc -l)
[ "$n_npz" -eq 8 ] || { echo "  ABORT: expected 8 axon1 shards, found $n_npz"; exit 1; }
ls -la cache/axon1_*.npz
$PY -u spc/verify_cache.py --cache cache/axon1 --expect axon1 --out $L/verify_axon1.json || exit 1

step "4/5 axon1: dedup vs testset in feature space, then evaluate raw AND testset-removed"
$PY -u spc/dedup_axon1.py --device cuda:0 --out $L/axon1_overlap.npz || exit 1
$PY -u spc/eval_axon1.py --ckpt $L/default.pt --device cuda:0 --out $L/axon1.json || exit 1

step "5/5 speed benchmark on an exclusive GPU"
$PY -u spc/bench_speed.py --ckpt $L/default.pt --device cuda:0 --out $L/bench.json || exit 1

echo; echo "############ $(date '+%F %T')  STAGE 3 COMPLETE ############"

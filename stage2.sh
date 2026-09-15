#!/usr/bin/env bash
# Everything that runs once the trainset feature cache exists. Sequential on purpose: each stage
# gates the next, and a failure must stop the chain rather than produce a report with a gap in it.
set -uo pipefail
cd /datasets/work/vLLM/temp/PAAS_simplicity
PY=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
export PYTHONNOUSERSITE=1
L=runs/spc
mkdir -p $L
step() { echo; echo "############ $(date '+%F %T')  $1 ############"; }

step "1/8 verify trainset cache (G2: ok[] + full duplicate scan + norms)"
$PY -u spc/verify_cache.py --cache cache/train --expect train --logs 'runs/extract/train_*.log' \
    --out $L/verify_train.json || exit 1

step "2/8 extract EVAL features (30,197) on 4 GPUs"
EV_PIDS=()
for s in 0 1 2 3; do
  # NOT setsid: setsid forks when the shell has made it a process-group leader, so $! would be the
  # setsid pid and `wait` would return IMMEDIATELY -- verify_cache would then run on a half-written
  # cache and "pass". Observed directly: a launch reported pid 2543874 while the python was 2543875.
  CUDA_VISIBLE_DEVICES=$s nohup $PY -u spc/extract_features.py --src manifests/eval.json \
     --out cache/eval --which-part $s --n-divided 4 --batch-size 64 --workers 8 \
     > runs/extract/eval_$s.log 2>&1 < /dev/null &
  EV_PIDS+=($!)
done
fail=0
for p in "${EV_PIDS[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "  eval extraction done, $fail shard(s) exited non-zero"
[ "$fail" -eq 0 ] || exit 1
$PY -u spc/verify_cache.py --cache cache/eval --expect eval --logs 'runs/extract/eval_*.log' \
    --out $L/verify_eval.json || exit 1

step "2b/8 precompute contamination artifacts (content dups, contradictory labels, max-cos to TRAIN)"
# Always run it: it is ~20 s, and "the file exists" is not evidence that it matches the current
# split.json, seed or grouping version. train_spc validates the fingerprint and will refuse a stale
# artifact anyway, so regenerating is strictly cheaper than debugging a refusal.
$PY -u spc/precompute_contam.py || exit 1

step "3/8 parity gates (G1 preprocessing bit-identity, G3 bf16 skew)"
$PY -u spc/verify_parity.py || exit 1

step "4/8 paper-faithful anchor (H0, T0, lr 2e-5, bs128, 2ep)"
$PY -u spc/train_spc.py --config configs/spc_paper.json --out $L/spc_paper.pt || exit 1

step "4b/8 G3 the gate that matters: does the extraction dtype move a reported metric?"
$PY -u spc/verify_dtype_impact.py --ckpt $L/spc_paper.pt --n 4096 || exit 1

step "5/8 selection sweep on DEV-A (prompts -> hparams -> heads -> imbalance)"
$PY -u spc/sweep.py --stage all --out $L/sweep.json || exit 1

step "6/8 train the DEV-A-selected default (and confirm once on DEV-B)"
$PY -u spc/train_spc.py --config configs/default.json --out $L/default.pt || exit 1

step "7/8 TESTSET evaluation (thresholds fitted on DEV-A, applied unchanged)"
$PY -u spc/evaluate.py --ckpt $L/default.pt --cache cache/eval --expect eval --fit-dev \
    --title "TESTSET mids_testset.json (n=30,197)" --save $L/testset.json || exit 1

step "8/8 attribution ladder: paper anchor on the same testset"
$PY -u spc/evaluate.py --ckpt $L/spc_paper.pt --cache cache/eval --expect eval --fit-dev \
    --title "TESTSET - PAPER RECIPE anchor" --save $L/testset_paper.json || exit 1

echo; echo "############ $(date '+%F %T')  STAGE 2 COMPLETE ############"

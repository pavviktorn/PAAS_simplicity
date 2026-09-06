#!/usr/bin/env bash
# PAAS_simplicity / PE-SPC -- full pipeline. Every stage is resumable: each extractor skips an
# existing output npz, so a re-run after a failure continues instead of redoing 8 GPU-hours.
#
# ONE venv for everything (same interpreter as every previous model in this project family):
#   /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
# -e matters here: without it a `fan` that lost every shard fell through to the verifier, which then
# happily certified an OLDER complete cache and the pipeline carried on as if the extraction had
# worked. Every stage below either succeeds or stops the run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python}
export PYTHONNOUSERSITE=1
GPUS=${GPUS:-0,1,2,3}
IFS=',' read -ra G <<< "$GPUS"
N=${#G[@]}
mkdir -p runs/extract runs/spc cache manifests

log() { echo "[$(date '+%F %T')] $*"; }

fan() {   # fan(script, extra_args..., nshards) -- one process per shard, round-robin over GPUS
  local script=$1; shift
  local nsh=$1; shift
  local pids=()
  for ((s=0; s<nsh; s++)); do
    local gpu=${G[$((s % N))]}
    # NOT setsid -- see stage2.sh: setsid forks and `wait $!` would return immediately.
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u "$script" --which-part "$s" --n-divided "$nsh" "$@" \
      > "runs/extract/$(basename "$script" .py)_$s.log" 2>&1 < /dev/null &
    pids+=($!)
  done
  log "launched ${#pids[@]} shards of $script on GPUS=$GPUS -> pids ${pids[*]}"
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  log "$script finished, $fail shard(s) exited non-zero"
  return $fail
}

case "${1:-all}" in
 train_feats)
   fan spc/extract_features.py "$N" --src manifests/train.json --out cache/train --batch-size 64 --workers 12
   $PY spc/verify_cache.py --cache cache/train --expect train --logs 'runs/extract/train_*.log' \
       --out runs/spc/verify_train.json ;;
 eval_feats)
   fan spc/extract_features.py "$N" --src manifests/eval.json --out cache/eval --batch-size 64 --workers 12
   $PY spc/verify_cache.py --cache cache/eval --expect eval --out runs/spc/verify_eval.json ;;
 contam)   # MUST run before any training: without it the de-contaminated DEV levels do not exist and
           # train_spc now refuses rather than reporting raw DEV numbers under the label "c99".
           $PY -u spc/precompute_contam.py ;;
 dtype)    # the G3 gate that actually matters -- needs a trained head, so it follows `paper`
           $PY -u spc/verify_dtype_impact.py --ckpt runs/spc/spc_paper.pt --n 4096 ;;
 axon1_feats)
   # 2 processes per GPU: the decode is single-threaded per process, so this overlaps CPU decode
   # with GPU encode instead of leaving one of the two idle.
   fan spc/extract_axon1.py "$((N*2))" --out cache/axon1 --batch-size 64
   nsh=$((N*2)); got=$(ls cache/axon1_*.npz 2>/dev/null | wc -l)
   [ "$got" -eq "$nsh" ] || { log "axon1: expected $nsh shards, found $got"; exit 1; }
   # --expect axon1, not none: 614,029 frames = 48,526 real + 565,503 pad. With --expect none a
   # PARTIAL 614k-frame decode passed verification.
   $PY spc/verify_cache.py --cache cache/axon1 --expect axon1 --out runs/spc/verify_axon1.json ;;
 parity)   $PY spc/verify_parity.py ;;
 split)    $PY spc/split_dev.py ;;
 protos)   PROTO_DEVICE=cpu $PY spc/text_prototypes.py ;;
 paper)    $PY spc/train_spc.py --config configs/spc_paper.json ;;
 sweep)    $PY -u spc/sweep.py --stage all --out runs/spc/sweep.json ;;
 default)  $PY spc/train_spc.py --config configs/default.json ;;
 test)     $PY spc/evaluate.py --ckpt runs/spc/default.pt --cache cache/eval --expect eval \
              --fit-dev --title "TESTSET mids_testset (30,197)" --save runs/spc/testset.json ;;
 axon1)    # eval_axon1.py, NOT evaluate.py: axon1 rows are video frames keyed "<path>#frame=N",
           # it has no deepfake class, it must be reported raw AND testset-removed, and its thresholds
           # must come from the checkpoint (fitted on DEV-A). Calling evaluate.py here self-fitted a
           # threshold on axon1 and printed it as deployable -- the advertised `all` path therefore
           # disagreed with stage3.sh, which was right.
           $PY spc/dedup_axon1.py --out runs/spc/axon1_overlap.npz
           $PY spc/eval_axon1.py --ckpt runs/spc/default.pt --overlap runs/spc/axon1_overlap.npz \
              --out runs/spc/axon1.json ;;
 bench)    $PY spc/bench_speed.py --ckpt runs/spc/default.pt ;;
 all)
   # ORDER: caches -> split -> prototypes -> contamination artifact -> gates -> paper anchor ->
   # dtype gate -> sweep -> default -> testset -> axon1 -> speed. contam must precede any training.
   for s in train_feats eval_feats split protos contam parity paper dtype sweep default test \
            axon1_feats axon1 bench; do
     log "=== STAGE $s ==="; bash "$0" "$s" || { log "STAGE $s FAILED"; exit 1; }
   done ;;
 *) echo "usage: $0 {train_feats|eval_feats|axon1_feats|contam|dtype|parity|split|protos|paper|sweep|default|test|axon1|bench|all}"; exit 2 ;;
esac

#!/usr/bin/env bash
# Exercise the three axon1 modules end-to-end on a handful of media files. These are the only modules
# never yet RUN (the earlier 3-media decode test was starved by the 48-worker extraction and killed),
# and they are the ones that run last in the real pipeline -- i.e. the worst place to discover a bug.
set -uo pipefail
cd /datasets/work/vLLM/temp/PAAS_simplicity
PY=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
export PYTHONNOUSERSITE=1
T=runs/axon1_test
mkdir -p $T
rm -f cache/axtest_*.npz

echo "#### decode + extract: 2 shards x a few media each ####"
P=()
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$((s+1)) AXON1_THREADS=3 AXON1_CV_THREADS=2 nohup $PY -u spc/extract_axon1.py \
    --out cache/axtest --which-part $s --n-divided 400 --limit-media 3 --batch-size 32 \
    > $T/extract_$s.log 2>&1 < /dev/null &
  P+=($!)
done
fail=0; for p in "${P[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "  extract shards done, $fail non-zero"
tail -2 $T/extract_0.log; tail -2 $T/extract_1.log
[ "$fail" -eq 0 ] || exit 1

echo; echo "#### verify the axon1-shaped cache (keys, not paths) ####"
$PY -u spc/verify_cache.py --cache cache/axtest --expect none --out $T/verify.json || exit 1

echo; echo "#### dedup vs the testset (sensitivity table + group-level mask) ####"
$PY -u spc/dedup_axon1.py --axon1 cache/axtest --eval cache/eval --device cuda:1 \
   --out $T/overlap.npz || exit 1

echo; echo "#### evaluate: raw + frame-level + group-level, vs FFAA on the same frames ####"
CK=runs/rehearsal/default.pt; [ -f $CK ] || CK=runs/rehearsal/spc_paper.pt
$PY -u spc/eval_axon1.py --ckpt $CK --cache cache/axtest --overlap $T/overlap.npz \
   --device cuda:1 --out $T/axon1.json || exit 1
echo; echo "#### axon1 chain OK ####"

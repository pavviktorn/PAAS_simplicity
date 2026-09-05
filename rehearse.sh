#!/usr/bin/env bash
# Dry rehearsal of the WHOLE chain at tiny compute. Its only job is to prove that every stage writes
# what the next stage reads and that report/build_exp24.py can render a PDF from the result -- the
# class of defect that only fires at the very end (an undefined name after 20 minutes of sweeping, or
# a key the report expects that nobody writes) is invisible to any amount of reading.
set -uo pipefail
cd /datasets/work/vLLM/temp/PAAS_simplicity
PY=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
export PYTHONNOUSERSITE=1
L=runs/rehearsal
mkdir -p $L
step(){ echo; echo "#### rehearse: $1 ####"; }

step "paper anchor (2% of TRAIN, 1 epoch)"
$PY - <<'PYEOF'
import json
d=json.load(open("configs/spc_paper.json")); d.update(train_frac=0.02, epochs=1, eval_every=0)
json.dump(d, open("/tmp/rehearse_paper.json","w"), indent=1)
PYEOF
$PY -u spc/train_spc.py --config /tmp/rehearse_paper.json --out $L/spc_paper.pt || exit 1

step "dtype-impact gate (uses the paper anchor head)"
# A 2%-data anchor cannot support a dtype verdict (see MIN_AUC in the gate). Exercising the code path
# is the point here; the real verdict comes from stage2, which runs it on the full-trainset anchor.
$PY -u spc/verify_dtype_impact.py --ckpt $L/spc_paper.pt --n 2048 --out $L/dtype_impact.json \
  || echo "  (expected for a 2% anchor: no verdict / fail -- path exercised, continuing)"

step "full sweep, all 4 stages, 2% of TRAIN"
$PY -u spc/sweep.py --stage all --train-frac 0.02 --epochs 1 --out $L/sweep.json --default-out $L/default_config.json || exit 1

step "train the selected default (2% override so the rehearsal stays cheap)"
$PY - <<'PYEOF'
import json
d=json.load(open("runs/rehearsal/default_config.json")); d["train_frac"]=0.02; d["epochs"]=1
json.dump(d, open("/tmp/rehearse_default.json","w"), indent=1)
print("  default.json as written by the sweep:", json.dumps(d))
PYEOF
$PY -u spc/train_spc.py --config /tmp/rehearse_default.json --out $L/default.pt || exit 1

step "testset evaluation + paper anchor on testset"
$PY -u spc/evaluate.py --ckpt $L/default.pt --cache cache/eval --expect eval --fit-dev \
   --title "REHEARSAL testset" --save $L/testset.json || exit 1
$PY -u spc/evaluate.py --ckpt $L/spc_paper.pt --cache cache/eval --expect eval --fit-dev \
   --title "REHEARSAL testset paper" --save $L/testset_paper.json || exit 1

step "speed benchmark (tiny iters, just to produce the artifact)"
$PY -u spc/bench_speed.py --ckpt $L/default.pt --device cuda:0 --batches 1,8 --iters 20 \
   --e2e-n 32 --out $L/bench.json || exit 1

step "copy the gates the report needs, then render the PDF to a THROWAWAY file"
cp runs/spc/verify_train.json runs/spc/parity.json runs/spc/contam_summary.json $L/ 2>/dev/null
SPC_RUNS=$PWD/$L EXPERIMENTS_PDF=/tmp/rehearsal_EXPERIMENTS.pdf $PY report/build_exp24.py || exit 1
echo; echo "#### rehearsal complete ####"

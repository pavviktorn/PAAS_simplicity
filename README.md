# PAAS_simplicity — PE-SPC, 3-class (real / pad / deepfake)

A 3,843-parameter detector: a **frozen** Perception Encoder (PE-Core-G14-448, 2.42 B params) plus
one trained prototype per class.

Implements **PE-SPC / Semantic Prototype Calibration** (arXiv 2608.04935) with the framing of
**"Simplicity Prevails"** (arXiv 2602.01738): a linear probe on a strong frozen vision foundation
model, where the probe is *initialised from the text tower* and only the prototypes train.

Both papers are **binary** (real vs AI-generated). This project is **3-class**, which is the part
with no paper to copy — the third prompt has to be chosen empirically, on a held-out split.

```
                 frozen, never updated                  trained (3,843 params)
  image ──► PE squash-resize 448 ──► PE vision ──► 1280-d ──► [3×1280 prototypes + 3 bias] ──► logits
                                                                        (+1 optional scale)
  fake_score = 1 − softmax(logits)[real]        decision: FAKE iff fake_score ≥ tau
```

## Layout

| path | what |
|---|---|
| `spc/extract_features.py` | frozen-feature extractor (sharded, 4 GPUs) → `cache/train_*.npz` |
| `spc/extract_axon1.py` | same, but decodes **video frames** for axon1, keyed `<path>#frame=NNNNNN` |
| `spc/verify_cache.py` | **G2 gate**: `ok[]`, class histogram, unit-norm, duplicate clusters |
| `spc/verify_parity.py` | **G1/G3 gates**: transform bit-identity, fp32-vs-bf16 feature cosine |
| `spc/split_dev.py` | group-disjoint TRAIN / DEV-A / DEV-B (stateless, path-derived) |
| `spc/text_prototypes.py` | embeds the 13 candidate prompt triplets + K=4 multi-prompt |
| `spc/head.py` | H0 linear (paper) · H1 +scale · H2 cosine · H3 multi-prototype · H4 MLP (diagnostic) |
| `spc/train_spc.py` | trains on the cache; fits thresholds on DEV-A; confirms once on DEV-B |
| `spc/sweep.py` | selection: prompts → hyperparameters → head → imbalance, all on DEV-A |
| `spc/evaluate.py` | full metric block with per-identity / per-attack-family breakdowns |
| `spc/eval_axon1.py` | axon1, raw **and** testset-removed, vs FFAA on the *same frames* |
| `spc/dedup_axon1.py` | feature-space near-duplicate detection (testset ⊂ axon1) |
| `spc/score_baselines.py` | mids9c / GSD / SeLop on the same testset, one harness |
| `spc/bench_speed.py` | GPU-only and end-to-end throughput/latency |
| `spc/infer.py` | single-image inference with the fingerprint assertions live |
| `run_all.sh`, `stage2.sh`, `stage3.sh` | the pipeline, resumable at every stage |

## Run

```bash
bash run_all.sh train_feats      # ~8 GPU-hours, once
bash stage2.sh                   # verify → eval feats → parity → paper anchor → sweep → default → testset
bash stage3.sh                   # baselines + axon1 (decode 614k frames) + speed benchmark
python report/build_exp24.py     # appends Experiment 24 to EXPERIMENTS.pdf
```

One interpreter for everything, matching every previous model in this project family:
`/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python` (with `PYTHONNOUSERSITE=1`).

## Things that will silently give you a wrong number

**PE-SPC must never be fed `paas/preprocess.py` output.** That is the project's canonical whole-frame
preprocessor and it **letterboxes** (pads to square). PE's own transform **squashes** (`Resize(448,448)`).
These are different pixels, there is no error, and the score just gets worse. Three train/serve skews
of exactly this shape already happened in this project family (A2 crop-vs-letterbox, GSD double-crop,
SeLop RandomResizedCrop-vs-Resize). There is exactly **one** transform constructor in this codebase —
`pt.get_image_transform(model.image_size)` — and every checkpoint stores the transform `repr`, image
size, dtype and the encoder's sha256, which `infer.py` asserts before it will score anything.

**`acc@0.5` is meaningless for this head.** With unit features, unit prototypes and bias `[0,1,1]` the
logits span `[-1, 2]`, so max softmax ≈ 0.44 — measured `rec3_real = 0.305` at `bin_auc = 1.000000`.
Reported accuracy uses the **deployable** rule instead: threshold the fake score, then argmax within
the two fake classes. Raw argmax is still printed, tagged `NOT-A-DECISION-METRIC`.

**Thresholds are fitted on DEV-A and applied unchanged**, with the *achieved* real recall printed next
to every target. A threshold fitted on the set it is reported on is not a result.

**Never `pgrep -f` / `pkill -f` a pattern here.** The pattern also matches the command line of the
shell doing the matching. In this project that killed a launcher mid-loop and produced two false
"still running" reports; `stage3.sh` waits on explicit PIDs.

**The cache is in strided shard order, not manifest order** (`shard_k = items[k::4]`, concatenated
`0,1,2,3`). Split membership is therefore recomputed from each row's path, never stored as an index.

**`manifests/eval.json` is block-ordered by class.** Any first-N subsample is single-class and makes
AUC undefined. All subsampling is strided and asserted to contain 3 classes.

## Data

| set | n | composition |
|---|---|---|
| train | 1,360,956 | real 426,901 / pad 551,945 / deepfake 382,110 |
| — TRAIN / DEV-A / DEV-B | 1,260,666 / 52,159 / 48,131 | 472,064 path-groups, 0 multi-class — but see the caveat below: path-grouping is **not** content-disjoint |
| testset (`mids_testset.json`) | 30,197 | real 10,104 / pad 10,368 / deepfake 9,725 |
| axon1 (held-out) | 614,029 | pad 565,503 / real 48,526 — **no deepfake class** |

**The validation split is contaminated and the testset is not.** The manifest holds 1,360,956 records
over 986,703 distinct paths but only **940,299 distinct feature vectors** — the same image is stored
under different paths (one cluster is 231 byte-identical copies of one real image). A path-keyed split
therefore cannot keep content on one side: **6.30%** of DEV-A's distinct images are bit-identical to a
training image (measured by hashing the feature rows; an earlier raw-fp16-dot threshold under-counted
this by ~40%), 11.35% are within cosine 0.99, and the *median* DEV-A image sits at 0.9565 to its
nearest training neighbour. This data is built from dense near-identical capture clusters, so no split
of it yields an independent validation set. 285 content clusters also carry contradictory labels
(276 pad+deepfake, 9 real+deepfake; 1,815 rows) and are excluded from every validation metric.

The testset is clean against the same trainset — **0 bit-identical, 0 within cosine 0.99, median ~0.83** — so the contamination affects *which config is selected*, not the reported result. Selection
is therefore made at three de-contamination levels (raw / ≥0.99 removed / ≥0.95 removed) and is only
trusted where the ranking is stable across them; see `spc/precompute_contam.py` and `spc/sweep.py`.

Labels come **only** from `get_label_all(path)` (`MAKEUP→PAD`, `UNKNOWN` dropped). The trainset json's
`cls_label` field is binary and disagrees with the path on PAD; it is never used as a label source.

axon1 rows are **video frames**, not files. They are decoded sequentially with `cv2` using the same
key convention as `PAAS_ensemble_v4/scripts/run_dataset.py`, so each feature joins to its recorded
baseline score *by key* — which is why this comparison does not carry the ~3 pp frame-alignment noise
the earlier axon1 comparisons had to.

The testset was **sampled from** axon1 and shares the 6 real identities and all 7 PAD families, so
axon1 is always reported twice: raw, and with the shared frames removed.

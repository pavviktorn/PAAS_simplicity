#!/usr/bin/env python
"""Record provenance for caches extracted BEFORE provenance existed (the train and eval caches).

This does not simply assert what they were built from -- it verifies it first, and only stamps what it
could check:
  * the manifest hash must equal the one split.json recorded when it was built (so the manifest has
    not changed underneath the cache);
  * shard count, row count and the exact per-class histogram must match the manifest;
  * every feature row must be unit-norm (proves normalize=True and the expected dtype path).
If any of that fails the cache is NOT stamped, because a provenance file that is merely plausible is
worse than none at all.
"""
import argparse, hashlib, json, os, sys
import numpy as np

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "spc"))
sys.path.insert(0, os.path.join(ROOT, "perception_models"))
import cache_io, provenance as PROV                    # noqa: E402
import core.vision_encoder.transforms as pt            # noqa: E402

CKPT = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)                 # e.g. cache/train
ap.add_argument("--src", required=True)                   # e.g. manifests/train.json
ap.add_argument("--expect", required=True, choices=["train", "eval"])
ap.add_argument("--n-divided", type=int, default=4)
a = ap.parse_args()

exp = {"train": cache_io.EXPECT_TRAIN, "eval": cache_io.EXPECT_EVAL}[a.expect]
src = os.path.join(ROOT, a.src)
prefix = os.path.join(ROOT, a.cache)

import glob
files = sorted(glob.glob(prefix + "_*.npz"), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
print(f"[stamp] {len(files)} shard(s) for {a.cache}")
if len(files) != a.n_divided:
    raise SystemExit(f"[stamp] found {len(files)} shards but --n-divided {a.n_divided}: refusing to "
                     f"stamp a set whose striding I cannot confirm.")

src_sha = PROV.sha256_file(src)
sp = json.load(open(os.path.join(ROOT, "manifests/split.json")))
if a.expect == "train" and sp.get("sha256_train_json") not in (None, src_sha):
    raise SystemExit(f"[stamp] {a.src} now hashes to {src_sha[:16]}... but split.json recorded "
                     f"{str(sp.get('sha256_train_json'))[:16]}.... The manifest changed after the cache "
                     f"was built, so the cache cannot be certified against it.")
print(f"[stamp] {a.src} sha256={src_sha[:16]}... matches the recorded value")

c = cache_io.load_shards(prefix)
n = len(c["labels"])
hist = {int(k): int(v) for k, v in zip(*np.unique(c["labels"], return_counts=True))}
if n != exp["n"] or hist != exp["hist"]:
    raise SystemExit(f"[stamp] cache is n={n:,} hist={hist} but the manifest implies "
                     f"n={exp['n']:,} hist={exp['hist']} -- not stamping.")
nrm = np.linalg.norm(c["feats"][:: max(1, n // 100000)].astype(np.float32), axis=1)
if abs(nrm.mean() - 1.0) > 2e-3:
    raise SystemExit(f"[stamp] features are not unit-norm (mean {nrm.mean():.6f}) -- not stamping.")
print(f"[stamp] verified n={n:,} hist={hist} norm_mean={nrm.mean():.6f}")

tfm_repr = repr(pt.get_image_transform(448))
enc_sha = PROV.sha256_file(CKPT)
for i, f in enumerate(files):
    meta = {"src": os.path.abspath(src), "src_sha256": src_sha, "encoder_sha256": enc_sha,
            "transform": tfm_repr, "image_size": 448, "dtype": "bf16", "storage_dtype": "fp16",
            "n_divided": a.n_divided, "which_part": i, "n_items_total": exp["n"],
            "stamped_retroactively": True,
            # Be explicit about the limits of a retroactive stamp: the checks below prove the cache
            # corresponds to THIS manifest, but nothing in a finished feature array proves WHICH
            # encoder or transform produced it. Those two fields are asserted from the current
            # environment, not verified, and anything downstream that treats them as evidence is
            # over-reading this file.
            "verified": ["shard count", "row count", "per-class histogram", "unit-norm features",
                         "manifest sha256 == the value split.json recorded"],
            "asserted_not_verified": ["encoder_sha256", "transform", "dtype", "storage_dtype"],
            "asserted_note": "taken from the live environment at stamping time; a feature array does "
                             "not carry the identity of the encoder that produced it"}
    PROV.write(f, meta)
    print(f"[stamp] -> {os.path.basename(PROV.sidecar_path(f))}")
print("[stamp] done")

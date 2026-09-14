#!/usr/bin/env python
"""Gate the feature cache BEFORE any training reads it.

This exists because of a specific failure already recorded in this project: a silent black-image
substitution turned 32% of a GSD anchor set into one constant vector while every log line stayed
healthy and no exception was raised. A constant-vector cluster is invisible in a loss curve and
looks like a hyperparameter effect in a metric. So: assert, don't hope.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache_io                                        # noqa: E402

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache/train")
ap.add_argument("--expect", default="train", choices=["train", "eval", "axon1", "none"])
ap.add_argument("--logs", default=None, help="glob of extraction logs to grep for UNREADABLE")
ap.add_argument("--out", default=None)
a = ap.parse_args()

c = cache_io.load_shards(os.path.join(ROOT, a.cache))
print(f"[verify] shards: {len(c['files'])}")
for f in c["files"]:
    z = np.load(f, allow_pickle=True)
    # Two cache schemas exist: the IMAGE caches (extract_features.py) carry `unreadable`, while the
    # AXON1 cache (extract_axon1.py) carries `n_decode_failed` and no `unreadable` at all, because its
    # rows are decoded video frames rather than files. Reading z["unreadable"] unconditionally was a
    # KeyError on the axon1 path -- and it would have fired only AFTER the 614k-frame decode.
    extra = (f"unreadable_field={int(z['unreadable'])}" if "unreadable" in z.files else
             f"decode_failed_media={int(z['n_decode_failed'])}" if "n_decode_failed" in z.files else
             "no per-shard failure field")
    print(f"   {os.path.basename(f):<16} n={len(z['labels']):>8,} "
          f"ok0={int((z['ok']==0).sum()):>5} {extra} "
          f"(ok[] is the authoritative substitution check either way)")
exp = {"train": cache_io.EXPECT_TRAIN, "eval": cache_io.EXPECT_EVAL,
       "axon1": cache_io.EXPECT_AXON1, "none": None}[a.expect]
info = cache_io.verify(c, exp, tag="verify")

# Duplicate scan on the FULL set (verify() only samples). What this is really looking for is a
# CONSTANT-VECTOR CLUSTER -- the signature of silent black-image substitution, which once turned 32%
# of a GSD anchor set into one vector with every log line healthy.
#
# But identical feature vectors are NOT by themselves evidence of that: this trainset contains flat
# photo pools (one directory holds 156,010 images) and byte-identical duplicate FILES are ordinary in
# scraped data. Two genuinely different images colliding on all 1280 fp16 values is not plausible, so
# a cluster means "the same image appears N times", which is a dataset property, not a bug.
#
# Therefore: the ok[] check above is the ABORT (it detects substitution exactly and directly), and
# this scan REPORTS. It only aborts when a cluster is both large in absolute terms and a meaningful
# share of the data, i.e. when it could actually move a prototype or a metric. A false abort here
# would block the whole pipeline over a duplicated stock photo.
n = len(c["labels"])
import hashlib
keys = np.array([hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in c["feats"]])
u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
dupmax = int(cnt.max())
info["unique_rows_full"] = float(len(u) / n)
info["largest_identical_cluster"] = dupmax
info["rows_in_dup_clusters"] = int(cnt[cnt > 1].sum() - (cnt > 1).sum())
print(f"[verify] FULL duplicate scan: unique={len(u):,}/{n:,} ({len(u)/n:.6f}) "
      f"largest_identical_cluster={dupmax} redundant_rows={info['rows_in_dup_clusters']:,}")
top = np.argsort(-cnt)[:5]
cross_shard_ok = 0
for r, gi in enumerate(top):
    if cnt[gi] < 2:
        break
    rows = np.flatnonzero(inv == gi)
    shards = sorted(set(int(x) for x in c["shard"][rows])) if "shard" in c else []
    same_path = len(set(c["paths"][rows].tolist())) == 1
    print(f"[verify]   cluster {r+1}: n={int(cnt[gi]):,} labels="
          f"{sorted(set(int(x) for x in c['labels'][rows]))} shards={shards} "
          f"same_path={same_path}")
    for i in rows[:2]:
        print(f"[verify]      {c['paths'][i]}")
    # The manifest duplicates 41,557 paths exactly 10x. Consecutive records are STRIDED across the 4
    # shards, so those copies were encoded by DIFFERENT GPUs in DIFFERENT processes. Identical fp16
    # vectors across shards is therefore a free determinism check on the whole extraction.
    if same_path and len(shards) > 1:
        cross_shard_ok += 1
info["cross_shard_identical_clusters"] = cross_shard_ok
if cross_shard_ok:
    print(f"[verify] DETERMINISM: {cross_shard_ok} of the top duplicate clusters are the SAME PATH "
          f"encoded on >1 shard (different GPU, different process) and produced BIT-IDENTICAL fp16 "
          f"features -- the extraction is reproducible across GPUs.")
DUP_ABS = int(os.environ.get("DUP_ABS", "5000"))
DUP_FRAC = float(os.environ.get("DUP_FRAC", "0.01"))
if dupmax > DUP_ABS and dupmax / n > DUP_FRAC:
    raise SystemExit(f"[verify] FATAL: {dupmax:,} identical feature vectors ({dupmax/n*100:.2f}% of the "
                     f"cache) -> large enough to move a prototype and a metric. If those paths above "
                     f"are distinct real images, this is a substituted/constant image, not a dataset "
                     f"duplicate.")
if dupmax > 50:
    print(f"[verify] NOTE {dupmax:,} identical vectors in the largest cluster. Not fatal (see above): "
          f"ok[]=={info['n_bad']} is the substitution check, and duplicate FILES are expected here.")

if a.logs:
    import glob, subprocess
    tot = 0
    for lg in sorted(glob.glob(a.logs)):
        k = sum(1 for ln in open(lg) if "UNREADABLE" in ln)
        tot += k
        print(f"[verify] {os.path.basename(lg)}: UNREADABLE lines={k}")
    info["unreadable_log_lines"] = tot
    if tot:
        raise SystemExit(f"[verify] FATAL: {tot} UNREADABLE lines in extraction logs")

print(f"[verify] PASS  {json.dumps({k: v for k, v in info.items() if k != 'hist'})}")
if a.out:
    json.dump(info, open(a.out, "w"), indent=1)
    print(f"[verify] -> {a.out}")

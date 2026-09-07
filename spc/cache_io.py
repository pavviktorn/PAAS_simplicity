"""Load the frozen-feature cache and the group-disjoint splits.

The cache is the whole experiment: 1.36 M x 1280 fp16 = 3.3 GB, which fits on one GPU, so a full
training run is a few seconds of matmuls and an honest sweep over prompts x heads x hyperparameters
becomes affordable. Everything downstream reads features ONLY through here.
"""
from __future__ import annotations
import glob, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
EXPECT_TRAIN = {"n": 1360956, "hist": {0: 426901, 1: 551945, 2: 382110}}
EXPECT_EVAL = {"n": 30197, "hist": {0: 10104, 1: 10368, 2: 9725}}
# axon1 = 614,029 OK-status frames from the recorded baseline run: 48,526 real + 565,503 pad and NO
# deepfake class. Parsed directly from results_ffaa.txt, so a partial decode is caught by the count.
EXPECT_AXON1 = {"n": 614029, "hist": {0: 48526, 1: 565503}}


def sha256(path, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(nbytes), b""):
            h.update(chunk)
    return h.hexdigest()


def load_shards(prefix, want_paths=True, extra=()):
    """Concatenate <prefix>_*.npz in shard order. Returns dict with feats/labels/ok/paths."""
    files = sorted(glob.glob(f"{prefix}_*.npz"), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        raise SystemExit(f"[cache] no shards at {prefix}_*.npz")
    # Refuse to concatenate shards that were built from different inputs, or an incomplete set.
    import provenance as _prov
    prov_info = _prov.validate_set(prefix, files)
    F, L, OK, P = [], [], [], []
    EX = {k: [] for k in extra}
    for f in files:
        z = np.load(f, allow_pickle=True)
        F.append(z["feats"]); L.append(z["labels"]); OK.append(z["ok"])
        for k in extra:
            if k not in z.files:
                raise SystemExit(f"[cache] {os.path.basename(f)} has no '{k}' array")
            EX[k].append(z[k])
        if want_paths:
            # image caches store `paths`; the axon1 cache stores `keys` ("<video>#frame=NNNNNN"),
            # because its rows are decoded VIDEO FRAMES and not files on disk. Both are row-aligned
            # identifiers, so accept either -- verify_cache and the duplicate-cluster report only need
            # "something printable that identifies row i".
            P.append(z["paths"] if "paths" in z.files else z["keys"])
    out = {"feats": np.concatenate(F), "labels": np.concatenate(L).astype(np.int64),
           "ok": np.concatenate(OK).astype(np.int64), "files": files, "prov": prov_info,
           "shard": np.concatenate([np.full(len(x), i, np.int16) for i, x in enumerate(L)])}
    if want_paths:
        out["paths"] = np.concatenate(P)
    for k in extra:
        out[k] = np.concatenate(EX[k])
    return out


def verify(c, expect=None, tag="cache", dup_thresh=0.9999, strict=True):
    """The G2 gate: a silent black-image substitution is invisible in every log line but shows up
    here as ok==0 rows and as duplicate constant feature vectors."""
    msgs, fatal = [], []
    n = len(c["labels"])
    n_bad = int((c["ok"] == 0).sum())
    msgs.append(f"n={n:,} unreadable(ok==0)={n_bad}")
    if n_bad and strict:
        fatal.append(f"{n_bad} unreadable images -> their features are ONE CONSTANT VECTOR")
    hist = {int(k): int(v) for k, v in zip(*np.unique(c["labels"], return_counts=True))}
    msgs.append(f"hist={hist}")
    if expect:
        if n != expect["n"]:
            fatal.append(f"count {n:,} != expected {expect['n']:,}")
        if hist != expect["hist"]:
            fatal.append(f"class histogram {hist} != expected {expect['hist']}")
    f = c["feats"]
    if not np.isfinite(f).all():
        fatal.append("non-finite features (fp16 overflow or a bad forward)")
    sub = f[:: max(1, n // 200000)].astype(np.float32)
    nrm = np.linalg.norm(sub, axis=1)
    msgs.append(f"||f||2 mean={nrm.mean():.6f} min={nrm.min():.6f} max={nrm.max():.6f}")
    if abs(nrm.mean() - 1.0) > 2e-3 or nrm.min() < 0.99 or nrm.max() > 1.01:
        fatal.append("features are not unit norm -> normalize=True was lost somewhere")
    # duplicate-vector scan: what a silent substitution looks like in feature space
    h = np.ascontiguousarray(f).view(np.uint8).reshape(n, -1)
    keys = np.array([hashlib.blake2b(h[i].tobytes(), digest_size=8).digest() for i in
                     range(0, n, max(1, n // 200000))])
    uniq = len(np.unique(keys, axis=0)) / len(keys)
    msgs.append(f"unique_rows={uniq:.6f} (on {len(keys):,} sampled)")
    # NOT fatal. This trainset manifest legitimately repeats 41,557 paths exactly 10x and 16 paths 16x
    # -- 1,360,956 records over 986,703 distinct images -- so a unique-row fraction near 0.73 is the
    # EXPECTED value, not evidence of a constant-vector cluster. The check that actually detects silent
    # black-image substitution is `ok == 0` above (exact, and it reads 0 here); the cluster-size gate
    # with a sane bound lives in verify_cache.py, which also prints the offending paths.
    if uniq < dup_thresh:
        msgs.append(f"NOTE duplicate rows below {dup_thresh}: expected here (duplicate manifest "
                    f"records); ok==0 count is the substitution check and it is {n_bad}")
    print(f"[{tag}] " + " | ".join(msgs), flush=True)
    if fatal:
        raise SystemExit("[%s] FATAL:\n" % tag + "".join(f"  - {m}\n" for m in fatal))
    return {"n": n, "n_bad": n_bad, "hist": hist, "unique_rows": float(uniq),
            "norm_mean": float(nrm.mean())}


def assert_same_encoder(cache_prov, ckpt_fingerprint=None, protos_fingerprint=None, tag="consumer"):
    """The cached features, the trained head and the text prototypes must all come from ONE encoder.
    Nothing downstream checked this: a head could be scored against features from a different encoder
    with no error and a plausible number."""
    got = (cache_prov or {}).get("encoder_sha256")
    for nm, other in (("checkpoint", ckpt_fingerprint), ("prototypes", protos_fingerprint)):
        want = (other or {}).get("encoder_sha256")
        if got and want and got != want:
            raise SystemExit(
                f"[{tag}] ENCODER MISMATCH: the feature cache was built with "
                f"{got[:16]}... but the {nm} records {want[:16]}.... These live in different embedding "
                f"spaces; refusing to combine them.")
        if got and want:
            print(f"[{tag}] encoder matches between cache and {nm} ({got[:16]}...)", flush=True)
        elif not got:
            print(f"[{tag}] NOTE cache has no encoder provenance; cannot cross-check the {nm}.",
                  flush=True)


def to_gpu(c, device="cuda:0"):
    """fp16 on device (3.3 GB for the trainset); batches are cast to fp32 at use."""
    return (torch.from_numpy(c["feats"]).to(device),
            torch.from_numpy(c["labels"]).to(device))

#!/usr/bin/env python
"""Measure how much of DEV is really the trainset, and write it as a fingerprinted artifact.

Two numerical points this file exists to get right:

1. EXACT duplicates are found by HASHING the feature rows, never by a cosine cut. The cache stores
   fp16 vectors that were L2-normalised in fp32 before the cast, so their fp16 norms sit in
   [0.9956, 1.0049] and a RAW DOT of two bit-identical rows equals ||v||^2 in [0.9913, 1.0098] -- not
   1. An earlier version thresholded raw dots at >=0.9999 and called that "bit-identical", which
   MISSED about 40% of the real duplicates: it reported 3.79%/4.59% for DEV-A/DEV-B where hashing
   gives 6.30% (2,326/36,937) and 7.87% (2,711/34,458).

2. NEAR-duplicate cosine is computed on vectors RE-NORMALISED in fp32, so the number is an actual
   cosine and not a norm-scaled dot.

The artifact carries fingerprints of everything its positional masks depend on -- the cache path
sequence, split.json, the seed, and the grouping-logic version -- so it cannot be silently reused
after any of them changes.
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import torch

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "spc"))
import cache_io, split_dev                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache/train")
ap.add_argument("--device", default=os.environ.get("CONTAM_DEVICE", "cuda:0"))
ap.add_argument("--out", default=os.path.join(ROOT, "cache/dev_contam.npz"))
ap.add_argument("--summary", default=os.path.join(ROOT, "runs/spc/contam_summary.json"))
a = ap.parse_args()
DEV = a.device

t0 = time.time()
c = cache_io.load_shards(os.path.join(ROOT, a.cache))
sp_path = os.path.join(ROOT, "manifests/split.json")
sp = json.load(open(sp_path))
paths, lab = c["paths"].tolist(), c["labels"]
split = split_dev.assign(paths, set(sp["big_groups"]), sp["seed"])
n = len(paths)
print(f"[contam] {n:,} rows loaded+split in {time.time()-t0:.0f}s")

seen, uniq = set(), np.zeros(n, bool)
for i, p in enumerate(paths):
    if p not in seen:
        seen.add(p); uniq[i] = True
print(f"[contam] distinct paths {int(uniq.sum()):,}")

h = np.array([hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in c["feats"]])
u, inv = np.unique(h, return_inverse=True)
print(f"[contam] distinct feature vectors {len(u):,} ({len(u)/n*100:.1f}% of rows)")

# contradictory-label content clusters -> unscoreable at every level
order = np.argsort(inv, kind="mergesort"); si, sl = inv[order], lab[order]
st = np.flatnonzero(np.r_[True, si[1:] != si[:-1]]); en = np.r_[st[1:], len(si)]
bad = {int(si[b]) for b, e in zip(st, en) if e - b > 1 and len({int(x) for x in sl[b:e]}) > 1}
conflict = np.isin(inv, list(bad))
print(f"[contam] contradictory-label clusters {len(bad):,} covering {int(conflict.sum()):,} rows")

# EXACT duplicates of a path-distinct TRAIN row, by hash identity
tr = np.flatnonzero((split == 0) & uniq)
tr_content = np.unique(inv[tr])
exact_dup = np.isin(inv, tr_content) & (split != 0)
print(f"[contam] exact (hash-identical) duplicates of TRAIN among DEV rows: {int(exact_dup.sum()):,}")

# TRUE cosine to the nearest path-distinct TRAIN row, renormalised in fp32
def l2(x):
    x = x.to(torch.float32)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


ftr = l2(torch.from_numpy(c["feats"][tr]).to(DEV)).to(torch.float16)
maxcos = np.full(n, -1.0, np.float32)
dev_rows = np.flatnonzero(split != 0)
fd = l2(torch.from_numpy(c["feats"][dev_rows]).to(DEV)).to(torch.float16)
for i in range(0, len(dev_rows), 2048):
    d = fd[i:i + 2048]
    m = torch.full((d.shape[0],), -1.0, device=DEV, dtype=torch.float16)
    for j in range(0, len(tr), 262144):
        m = torch.maximum(m, (d @ ftr[j:j + 262144].t()).max(1).values)
    maxcos[dev_rows[i:i + 2048]] = m.float().cpu().numpy()
print(f"[contam] true cosine computed for {len(dev_rows):,} DEV rows "
      f"(max observed {maxcos[dev_rows].max():.6f} -- should be ~1.0, not >1.01)")

summary = {"n_records": int(n), "n_distinct_paths": int(uniq.sum()), "n_distinct_content": int(len(u)),
           "n_conflict_clusters": int(len(bad)), "n_conflict_rows": int(conflict.sum()),
           "exact_dup_method": "blake2b over the fp16 feature bytes (NOT a cosine threshold)",
           "cosine_method": "re-normalised in fp32 before the dot product"}


def profile(name, sel):
    cos = maxcos[sel]
    d = {"n": int(sel.sum()),
         "median_max_cos": float(np.median(cos)) if sel.any() else float("nan"),
         "frac_exact_dup": float(exact_dup[sel].mean()) if sel.any() else float("nan"),
         "frac_ge_099": float((cos >= 0.99).mean()) if sel.any() else float("nan"),
         "frac_ge_095": float((cos >= 0.95).mean()) if sel.any() else float("nan")}
    summary[name] = d
    print(f"[contam] {name}: n={d['n']:,} median_cos={d['median_max_cos']:.4f} "
          f"exact_dup={d['frac_exact_dup']*100:.2f}% >=0.99={d['frac_ge_099']*100:.2f}% "
          f">=0.95={d['frac_ge_095']*100:.2f}%")


for nm, k in (("devA", 1), ("devB", 2)):
    profile(nm, (split == k) & uniq & ~conflict)

# testset control against the same TRAIN set
ev_prefix = os.path.join(ROOT, "cache/eval")
if glob.glob(ev_prefix + "_*.npz"):
    ev = cache_io.load_shards(ev_prefix)
    he = np.array([hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in ev["feats"]])
    tr_hashes = set(h[tr].tolist())
    ex_t = np.array([x in tr_hashes for x in he.tolist()])
    fe = l2(torch.from_numpy(ev["feats"]).to(DEV)).to(torch.float16)
    bt = np.empty(len(fe), np.float32)
    for i in range(0, len(fe), 2048):
        d = fe[i:i + 2048]
        m2 = torch.full((d.shape[0],), -1.0, device=DEV, dtype=torch.float16)
        for j in range(0, len(tr), 262144):
            m2 = torch.maximum(m2, (d @ ftr[j:j + 262144].t()).max(1).values)
        bt[i:i + 2048] = m2.float().cpu().numpy()
    summary["testset"] = {"n": int(len(bt)), "median_max_cos": float(np.median(bt)),
                          "frac_exact_dup": float(ex_t.mean()),
                          "frac_ge_099": float((bt >= 0.99).mean()),
                          "frac_ge_095": float((bt >= 0.95).mean())}
    print(f"[contam] testset: n={len(bt):,} median_cos={np.median(bt):.4f} "
          f"exact_dup={ex_t.mean()*100:.3f}% >=0.99={(bt>=0.99).mean()*100:.2f}% "
          f">=0.95={(bt>=0.95).mean()*100:.2f}%")
    np.save(os.path.join(ROOT, "runs/spc/testset_traincos.npy"), bt)

# ---- fingerprints: everything these positional masks depend on --------------------------------
fp = {"paths_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
      "split_json_sha256": cache_io.sha256(sp_path),
      "split_seed": int(sp["seed"]),
      "groupkey_version": int(getattr(split_dev, "GROUPKEY_VERSION", 0)),
      "cache_prefix": a.cache,
      "cache_files": [os.path.basename(f) for f in c["files"]],
      "encoder_sha256": (c.get("prov") or {}).get("encoder_sha256")}
print(f"[contam] fingerprint paths={fp['paths_sha256'][:12]}... split.json={fp['split_json_sha256'][:12]}... "
      f"seed={fp['split_seed']} groupkey_v={fp['groupkey_version']}")

# content_id lets the trainer build any content-level mask it needs (e.g. one row per distinct image
# inside TRAIN) without re-hashing 1.36 M rows on every run.
np.savez(a.out, max_cos_to_train=maxcos, exact_dup_in_train=exact_dup, conflict=conflict,
         uniq_path=uniq, split=split.astype(np.int8), content_id=inv.astype(np.int32),
         fingerprint=np.array(json.dumps(fp)))
summary["fingerprint"] = fp
os.makedirs(os.path.dirname(a.summary), exist_ok=True)
json.dump(summary, open(a.summary, "w"), indent=1)
print(f"[contam] -> {a.out}\n[contam] -> {a.summary} ({time.time()-t0:.0f}s)")

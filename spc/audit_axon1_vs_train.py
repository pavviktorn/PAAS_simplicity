#!/usr/bin/env python
"""Is axon1 clean with respect to PE-SPC's TRAINSET?

axon1 is being treated as the one valid cross-model comparison, because every previous model in this
family selected on mids_testset.json while PE-SPC did not. That argument only holds if axon1 is ALSO
held out from PE-SPC's own training data -- otherwise the fair-comparison claim just moves the bias to
the other side of the table.

The same audit that was run on the testset (0 exact duplicates, 0 within cosine 0.99, median 0.831)
is therefore run here: exact duplicates by feature hash, and true cosine on fp32-renormalised vectors.
Reported per class, because axon1's reals are 6 video identities that the testset also draws from.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import torch

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "spc"))
import cache_io, split_dev                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default=os.path.join(ROOT, "runs/spc/axon1_vs_train.json"))
a = ap.parse_args()

tr = cache_io.load_shards(os.path.join(ROOT, "cache/train"))
sp = json.load(open(os.path.join(ROOT, "manifests/split.json")))
split = split_dev.assign(tr["paths"].tolist(), set(sp["big_groups"]), sp["seed"])
seen, uniq = set(), np.zeros(len(split), bool)
for i, p in enumerate(tr["paths"].tolist()):
    if p not in seen:
        seen.add(p); uniq[i] = True
keep = np.flatnonzero((split == 0) & uniq)               # path-distinct TRAIN rows only
print(f"[audit] TRAIN reference: {len(keep):,} path-distinct rows")

ax = cache_io.load_shards(os.path.join(ROOT, "cache/axon1"), want_paths=False, extra=("keys",))
lab, keys = ax["labels"].astype(np.int64), ax["keys"]
print(f"[audit] axon1: {len(lab):,} frames (real {int((lab==0).sum()):,} / pad {int((lab==1).sum()):,})")

htr = set(hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in tr["feats"][keep])
hax = [hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in ax["feats"]]
exact = np.array([h in htr for h in hax])
print(f"[audit] exact (hash-identical) duplicates of TRAIN: {int(exact.sum()):,} "
      f"({exact.mean()*100:.4f}%)")


def l2(x):
    x = x.to(torch.float32)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


D = a.device
ftr = l2(torch.from_numpy(tr["feats"][keep]).to(D)).to(torch.float16)
fax = l2(torch.from_numpy(ax["feats"]).to(D)).to(torch.float16)
best = np.empty(len(fax), np.float32)
for i in range(0, len(fax), 4096):
    d = fax[i:i + 4096]
    m = torch.full((d.shape[0],), -1.0, device=D, dtype=torch.float16)
    for j in range(0, len(keep), 262144):
        m = torch.maximum(m, (d @ ftr[j:j + 262144].t()).max(1).values)
    best[i:i + 4096] = m.float().cpu().numpy()

out = {"n": int(len(lab)), "frac_exact_dup": float(exact.mean()),
       "median_max_cos": float(np.median(best))}
print(f"[audit] max-cos-to-TRAIN percentiles: "
      + " ".join(f"p{p}={np.percentile(best,p):.4f}" for p in (25, 50, 75, 90, 99)))
for t in (0.9999, 0.99, 0.98, 0.95, 0.90):
    out[f"frac_ge_{t}"] = float((best >= t).mean())
    print(f"[audit]   >= {t:<7}: {int((best>=t).sum()):>8,} ({(best>=t).mean()*100:6.3f}%)")
for cl, nm in ((0, "real"), (1, "pad")):
    m = lab == cl
    if m.any():
        out[nm] = {"n": int(m.sum()), "median_max_cos": float(np.median(best[m])),
                   "frac_exact_dup": float(exact[m].mean()),
                   "frac_ge_099": float((best[m] >= 0.99).mean()),
                   "frac_ge_095": float((best[m] >= 0.95).mean())}
        print(f"[audit]   {nm:<5} n={int(m.sum()):>8,} median={np.median(best[m]):.4f} "
              f"exact={exact[m].mean()*100:.4f}% >=0.99={(best[m]>=0.99).mean()*100:.3f}% "
              f">=0.95={(best[m]>=0.95).mean()*100:.3f}%")
ts = json.load(open(os.path.join(ROOT, "runs/spc/contam_summary.json"))).get("testset", {})
print(f"\n[audit] for reference, the TESTSET against the same TRAIN reference: "
      f"exact={ts.get('frac_exact_dup',float('nan'))*100:.4f}% "
      f">=0.99={ts.get('frac_ge_099',float('nan'))*100:.3f}% "
      f"median={ts.get('median_max_cos',float('nan')):.4f}")
out["testset_reference"] = ts
np.save(a.out.replace(".json", "_cos.npy"), best)
json.dump(out, open(a.out, "w"), indent=1)
print(f"[audit] -> {a.out}")

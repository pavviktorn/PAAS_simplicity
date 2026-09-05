#!/usr/bin/env python
"""Merge sharded baseline score files and recompute the metric block over the union."""
import glob, json, sys, os
import numpy as np
sys.path.insert(0, "/datasets/work/vLLM/temp/PAAS_simplicity/spc")
import metrics as M

pat, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(pat))
if not files:
    raise SystemExit(f"no shards matching {pat}")
img, lab, sc = [], [], {}
members = None
for f in files:
    d = json.load(open(f))
    members = members or d["members"]
    if d["members"] != members:
        raise SystemExit(f"{f} scored {d['members']} but another shard scored {members}")
    img += d["image"]; lab += d["label"]
    for k in members:
        sc.setdefault(k, []).extend(d["scores"][k])
n_before = len(img)
seen, keep = set(), []
for i, p in enumerate(img):
    if p not in seen:
        seen.add(p); keep.append(i)
if len(keep) != n_before:
    print(f"[merge] dropped {n_before-len(keep):,} duplicate rows across shards")
lab = np.array(lab)[keep]
print(f"[merge] {len(files)} shard(s) -> {len(keep):,} images | classes {np.bincount(lab).tolist()}")
summary = {}
for k in members:
    v = np.array(sc[k], dtype=np.float64)[keep]
    ok = np.isfinite(v)
    m = M.block(v[ok], lab[ok])
    summary[k] = m
    print(f"  {k:<16} n={int(ok.sum()):>6,} bin_auc={m['bin_auc']:.6f} ap={m['ap']:.6f} "
          f"eer={m['eer']:.6f} fr@98={m['fake_rec@real98']:.6f} "
          f"pad={m['pad_rec@real98']:.6f} df={m['deepfake_rec@real98']:.6f}")
json.dump({"n": len(keep), "members": members, "summary": summary,
           "scores": {k: np.array(sc[k])[keep].tolist() for k in members},
           "label": lab.tolist(), "image": [img[i] for i in keep],
           "merged_from": [os.path.basename(f) for f in files]}, open(out, "w"), indent=1)
print(f"[merge] -> {out}")

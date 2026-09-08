#!/usr/bin/env python
"""Find axon1 frames that are also in the testset -- in feature space, and honestly.

WHY THIS IS NOT A ONE-THRESHOLD PROBLEM (measured, 2026-08-20):

  same image, JPEG re-encode q=98 : cosine mean 0.9878  min 0.9388
  same image, JPEG re-encode q=90 : cosine mean 0.9748  min 0.8528
  same image, resize x0.5         : cosine mean 0.9843  min 0.9319
  DIFFERENT images                : cosine max  0.8740  p99 0.8004

The two bands OVERLAP. A JPEG round-trip can move a PE feature further than two different images sit
apart -- and the testset .jpg files ARE re-encodes of axon1 video frames. So the 0.999 threshold this
script originally used would have matched almost nothing and produced a "testset-removed" number that
removed nothing while claiming otherwise. That is precisely the class of silent failure this project
keeps hitting.

Two removals are therefore produced, and both get reported:

  FRAME-level  cos >= --thresh (default 0.95, calibrated above): catches the re-encoded frame itself.
  GROUP-level  every frame of any MEDIA FILE that contributed a frame-level match. This is the
               conservative one and it is the number to trust: adjacent frames of one clip are
               near-duplicates of each other, so removing only the matched frame still leaves its
               neighbours behind -- exactly the leak the group-disjoint train split exists to prevent.

The full threshold sensitivity table is printed so the choice is auditable rather than asserted.
"""
import argparse, collections, glob, hashlib, json, os, sys
import numpy as np
import torch

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "spc"))
import cache_io                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--axon1", default="cache/axon1")
ap.add_argument("--eval", default="cache/eval")
ap.add_argument("--thresh", type=float, default=0.95)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default=os.path.join(ROOT, "runs/spc/axon1_overlap.npz"))
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(ROOT, a.axon1) + "_*.npz"),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
FA, LA, KA = [], [], []
for f in files:
    z = np.load(f, allow_pickle=True)
    FA.append(z["feats"]); LA.append(z["labels"]); KA.append(z["keys"])
fa_np = np.concatenate(FA); lab = np.concatenate(LA).astype(np.int64)
keys = np.concatenate(KA)
E = cache_io.load_shards(os.path.join(ROOT, a.eval))
print(f"[dedup] axon1={fa_np.shape} eval={E['feats'].shape}")

fe = torch.from_numpy(E["feats"]).to(a.device).float()
lab_e = E["labels"]
best = np.empty(len(fa_np), np.float32)
best_j = np.empty(len(fa_np), np.int64)
CH = 8192
for i in range(0, len(fa_np), CH):
    fa = torch.from_numpy(fa_np[i:i + CH]).to(a.device).float()
    mx = (fa @ fe.t()).max(1)
    best[i:i + CH] = mx.values.cpu().numpy()
    best_j[i:i + CH] = mx.indices.cpu().numpy()
    del fa

pct = [1, 25, 50, 75, 90, 95, 99, 99.9]
print("[dedup] max-cosine-to-testset percentiles: "
      + " ".join(f"p{p}={v:.5f}" for p, v in zip(pct, np.percentile(best, pct))))

# media file per axon1 frame (strip the #frame= suffix) -> group-level removal
media = np.array([k.split("#frame=")[0] for k in keys])
uniq_media, media_idx = np.unique(media, return_inverse=True)
print(f"[dedup] {len(uniq_media):,} distinct media files across {len(keys):,} frames")

print(f"[dedup] {'thresh':>8} {'frames':>10} {'%':>7} {'media hit':>10} {'group frames':>13} {'%':>7}")
table = {}
for t in (0.9999, 0.999, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.88):
    fm = best >= t
    hit_media = np.unique(media_idx[fm])
    gm = np.isin(media_idx, hit_media)
    table[t] = {"frames": int(fm.sum()), "media": int(len(hit_media)), "group_frames": int(gm.sum())}
    print(f"[dedup] {t:>8.4f} {int(fm.sum()):>10,} {fm.mean()*100:>6.2f}% {len(hit_media):>10,} "
          f"{int(gm.sum()):>13,} {gm.mean()*100:>6.2f}%")

frame_mask = best >= a.thresh
hit_media = np.unique(media_idx[frame_mask])
group_mask = np.isin(media_idx, hit_media)
print(f"\n[dedup] CHOSEN thresh={a.thresh}")
print(f"[dedup]   FRAME-level removal: {int(frame_mask.sum()):,} frames "
      f"(real {int((frame_mask & (lab == 0)).sum()):,} / pad {int((frame_mask & (lab == 1)).sum()):,})")
print(f"[dedup]   GROUP-level removal: {int(group_mask.sum()):,} frames from "
      f"{len(hit_media):,}/{len(uniq_media):,} media "
      f"(real {int((group_mask & (lab == 0)).sum()):,} / pad {int((group_mask & (lab == 1)).sum()):,})")
left = ~group_mask
print(f"[dedup]   GROUP-level LEAVES {int(left.sum()):,} frames "
      f"(real {int((left & (lab == 0)).sum()):,} / pad {int((left & (lab == 1)).sum()):,})")
if int((left & (lab == 0)).sum()) < 500 or int((left & (lab == 1)).sum()) < 500:
    print("[dedup]   WARNING group-level removal leaves too little of one class to measure. The testset "
          "was drawn from across these media, so a clip-level de-contamination of axon1 is not "
          "achievable; the frame-level number must then be reported WITH that caveat, not as clean.")

# which testset classes are matching, as a sanity check: axon1 has no deepfake, so a deepfake match
# would mean the threshold is matching unrelated content
mc = collections.Counter(int(lab_e[j]) for j in best_j[frame_mask])
print(f"[dedup]   matched testset rows by class (0=real,1=pad,2=deepfake): {dict(mc)}")
if mc.get(2, 0) > 0.05 * max(sum(mc.values()), 1):
    print("[dedup]   WARNING >5% of matches point at testset DEEPFAKE rows, which do not exist in "
          "axon1 -> the threshold is matching unrelated content, not duplicates.")

# Tie the mask to the exact frame set and the exact testset it was computed against: it is applied
# positionally, so a stale same-length mask would otherwise remove the wrong frames.
axon_sha = hashlib.sha256("\n".join(keys.tolist()).encode()).hexdigest()
eval_sha = hashlib.sha256(np.ascontiguousarray(E["feats"]).tobytes()).hexdigest()
np.savez(a.out, best_cos=best, overlap=frame_mask, group_overlap=group_mask,
         thresh=a.thresh, table=json.dumps(table),
         axon1_keys_sha256=np.array(axon_sha), eval_feats_sha256=np.array(eval_sha))
print(f"[dedup] alignment: axon1_keys sha={axon_sha[:16]}... eval_feats sha={eval_sha[:16]}...")
print(f"[dedup] -> {a.out}")

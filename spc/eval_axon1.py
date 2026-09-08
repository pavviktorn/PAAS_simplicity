#!/usr/bin/env python
"""Evaluate PE-SPC on axon1, against FFAA measured on THE SAME FRAMES.

Two things make this comparison stronger than the previous axon1 comparisons in this project:

1. The recorded FFAA fake score travels with every frame KEY in the cache (extract_axon1.py joins on
   "<path>#frame=NNNNNN"), so FFAA's metrics are recomputed here on exactly the frames PE-SPC scored.
   Nothing is quoted from an aggregate over a possibly-different frame set -- which is what forced the
   earlier "~3 pp frame-alignment noise" caveat.

2. The testset was SAMPLED FROM axon1 and shares the 6 real identities and all 7 PAD families, so
   "it also generalises to axon1" is circular unless the shared frames come out. They are removed by
   feature-space near-duplicate detection (dedup_axon1.py) and BOTH numbers are reported.

axon1 has NO deepfake class: 565,503 pad + 48,526 real. This is a real-vs-PAD benchmark and is
reported as one; a 3-class accuracy here would have an empty cell.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hashlib                                          # noqa: E402
import cache_io, metrics as M                          # noqa: E402
import train_spc as T                                  # noqa: E402
from evaluate import load_head                         # noqa: E402

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
FAMILIES = ["Replay_mobile", "Replay_PC", "Textile", "Silicone", "latex", "Advanced", "Wrapped"]
REAL_IDS = ["id_R_10", "id_R_12", "id_R_13", "id_R_15", "id_R_16", "id_R_17"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/spc/default.pt"))
ap.add_argument("--cache", default="cache/axon1")
ap.add_argument("--overlap", default=os.path.join(ROOT, "runs/spc/axon1_overlap.npz"))
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default=os.path.join(ROOT, "runs/spc/axon1.json"))
a = ap.parse_args()

head, ck = load_head(a.ckpt, a.device)
tau = {int(k): float(v) for k, v in ck.get("tau", {}).items()}
print(f"[axon1] ckpt={a.ckpt} cfg={ck['cfg'].get('name')} params={head.n_trainable():,}")
print(f"[axon1] thresholds fitted on DEV-A (trainset split), applied here UNCHANGED: {tau}")

# Use the SHARED verified loader rather than a hand-rolled glob: it enforces provenance consistency,
# a complete shard set, the expected 614,029 frames = 48,526 real + 565,503 pad, unit-norm features and
# the ok[] substitution check. The manual version skipped all of that.
c = cache_io.load_shards(os.path.join(ROOT, a.cache), want_paths=False,
                         extra=("keys", "baseline"))
cache_io.verify(c, cache_io.EXPECT_AXON1, tag="axon1")
cache_io.assert_same_encoder(c.get("prov"), ck.get("fingerprint"), None, tag="axon1")
feats = c["feats"]; labels = c["labels"].astype(np.int64)
keys = c["keys"]; base = c["baseline"].astype(np.float64)
print(f"[axon1] {len(labels):,} frames | real {int((labels==0).sum()):,} pad {int((labels==1).sum()):,} "
      f"deepfake {int((labels==2).sum()):,} (axon1 has no deepfake by construction)")

fs, pr = T.score(head, torch.from_numpy(feats).to(a.device))

ov = None
if os.path.exists(a.overlap):
    z = np.load(a.overlap, allow_pickle=False)
    # The mask is POSITIONAL over axon1 rows. A same-length mask from an earlier decode (different
    # frame order, or a different eval cache on the other side of the comparison) would apply silently.
    want_keys = hashlib.sha256("\n".join(keys.tolist()).encode()).hexdigest()
    got_keys = str(z["axon1_keys_sha256"]) if "axon1_keys_sha256" in z.files else None
    if got_keys is None:
        raise SystemExit(f"[axon1] {a.overlap} predates the alignment checksum. Re-run "
                         f"spc/dedup_axon1.py so the mask can be tied to these exact frames.")
    if got_keys != want_keys:
        raise SystemExit(f"[axon1] {a.overlap} was built for a different axon1 frame set "
                         f"({got_keys[:16]}... vs {want_keys[:16]}...). Its mask is positional, so "
                         f"applying it would remove the wrong frames. Re-run spc/dedup_axon1.py.")
    if len(z["overlap"]) != len(labels):
        raise SystemExit(f"[axon1] overlap mask length {len(z['overlap']):,} != {len(labels):,} rows")
    ov = z["overlap"].astype(bool)
    gov = z["group_overlap"].astype(bool) if "group_overlap" in z.files else None
    print(f"[axon1] testset overlap: FRAME-level {int(ov.sum()):,} ({ov.mean()*100:.3f}%) at cos >= "
          f"{float(z['thresh'])}" + (f" | GROUP-level {int(gov.sum()):,} ({gov.mean()*100:.3f}%)"
                                     if gov is not None else ""))
else:
    gov = None
    print(f"[axon1] WARNING no overlap mask at {a.overlap} -- the 'testset-removed' number cannot be "
          f"produced, and the raw number alone is CIRCULAR (the testset was sampled from axon1).")

out = {"n": int(len(labels)), "tau": tau, "views": {}}


def view(key, name, sel):
    n = int(sel.sum())
    if n == 0 or len(set(labels[sel].tolist())) < 2:
        print(f"\n=== {name}: skipped (n={n}, needs both classes) ===")
        return
    print(f"\n=== {name} (n={n:,}: real {int((labels[sel]==0).sum()):,} / "
          f"pad {int((labels[sel]==1).sum()):,}) ===")
    rows = {}
    for who, sc in (("PE-SPC", fs), ("FFAA Qwen3.5-4B (same frames)", base)):
        m = M.block(sc[sel], labels[sel])
        rows[who] = m
        print(f"  {who:<32} bin_auc={m['bin_auc']:.6f} ap={m['ap']:.6f} eer={m['eer']:.6f}")
        for t in (95, 98, 99):
            print(f"     self-fit tau@real{t}={m[f'tau@real{t}']:.6f} achieved_real={m[f'real_rec@real{t}']:.6f} "
                  f"fake_rec={m[f'fake_rec@real{t}']:.6f}")
    # PE-SPC at the DEV-A thresholds, applied unchanged -- the only deployable number here
    for t, tv in sorted(tau.items()):
        real = fs[sel & (labels == 0)]; fake = fs[sel & (labels != 0)]
        print(f"  PE-SPC @ DEV-A tau@real{t}={tv:.6f} -> ACHIEVED real_rec={float((real<tv).mean()):.6f} "
              f"fake_rec={float((fake>=tv).mean()):.6f}")
        rows.setdefault("PE-SPC_devA_tau", {})[str(t)] = {
            "tau": tv, "real_rec": float((real < tv).mean()), "fake_rec": float((fake >= tv).mean())}
    out["views"][key] = {"title": name, "n": n,
                         "n_real": int((labels[sel] == 0).sum()),
                         "n_pad": int((labels[sel] == 1).sum()), **rows}


all_sel = np.ones(len(labels), bool)
view("raw", "AXON1 RAW (includes frames the testset was sampled from)", all_sel)
if ov is not None:
    view("frame_removed", "AXON1 FRAME-LEVEL testset-removed", ~ov)
if gov is not None and int((~gov).sum()):
    view("group_removed",
         "AXON1 GROUP-LEVEL testset-removed (conservative; whole media files dropped)", ~gov)

# Per-family PAD recall and per-identity real recall, at MATCHED REAL RECALL for both models.
#
# The previous version compared PE-SPC at its DEV-A tau@real99 against "FFAA@0.5" -- a fixed 0.5 for
# FFAA and a 99%-real-recall threshold for PE-SPC. Those are not the same operating point: on axon1
# PE-SPC's tau@real99 achieves 99.998% real recall, so its fake recall is necessarily low, while 0.5
# is an arbitrary point on FFAA's scale. That comparison made PE-SPC look far worse than a matched one
# does, and a per-family table is exactly where such a mismatch is invisible to the reader.
#
# Both models are therefore thresholded at the SAME achieved real recall, each fitted on the view being
# reported, so a difference in a family is a difference in detection rather than in calibration.
sel = (~ov) if ov is not None else all_sel
# 0.98, not 0.99. At 0.99 the comparison is DEGENERATE for FFAA: 1.201% of its REAL frames on this view
# score exactly 1.0000, the same value as 69.3% of the PAD frames, so no threshold can reach 99% real
# recall and its fake recall is mechanically 0. Reporting that as a per-family result would look like
# "FFAA detects nothing", which is false -- at 98% real recall it reaches 0.8525 on this view. The
# saturation itself is the finding, and it is reported separately below rather than smuggled into a
# recall column.
TARGET = float(os.environ.get("AXON1_MATCH_TARGET", "0.98"))
out["per_family_target_real_recall"] = TARGET


def tau_for(scores, target, m_real):
    t, ach = M.tau_at_real_recall(scores[m_real], target)
    return t, ach


m_real = sel & (labels == 0)
t_spc, a_spc = tau_for(fs, TARGET, m_real)
t_ffa, a_ffa = tau_for(base, TARGET, m_real)
print(f"\n-- matched operating point on this view: real recall {TARGET:.2f} --")
print(f"   PE-SPC tau={t_spc:.6f} (achieved real {a_spc:.6f}) | FFAA tau={t_ffa:.6f} "
      f"(achieved real {a_ffa:.6f})")
out["matched_tau"] = {"pe_spc": {"tau": float(t_spc), "real_rec": float(a_spc)},
                      "ffaa": {"tau": float(t_ffa), "real_rec": float(a_ffa)}}
print(f"-- per-PAD-family recall at that matched point --")
fam = {}
for f in FAMILIES:
    m = np.array([f in k for k in keys]) & (labels == 1) & sel
    if m.any():
        r_spc = float((fs[m] >= t_spc).mean()); r_ffaa = float((base[m] >= t_ffa).mean())
        fam[f] = {"n": int(m.sum()), "pe_spc": r_spc, "ffaa": r_ffaa, "delta": r_spc - r_ffaa}
        print(f"     {f:<16} n={int(m.sum()):>7,} PE-SPC={r_spc:.6f}  FFAA={r_ffaa:.6f}  "
              f"delta={r_spc-r_ffaa:+.6f}")
out["pad_families"] = fam
print(f"-- per-identity REAL recall at that matched point --")
ids = {}
for i in REAL_IDS:
    m = np.array([i in k for k in keys]) & (labels == 0) & sel
    if m.any():
        r_spc = float((fs[m] < t_spc).mean()); r_ffaa = float((base[m] < t_ffa).mean())
        ids[i] = {"n": int(m.sum()), "pe_spc": r_spc, "ffaa": r_ffaa}
        print(f"     {i:<16} n={int(m.sum()):>7,} PE-SPC={r_spc:.6f}  FFAA={r_ffaa:.6f}")
ph = np.array(["photo_reals" in k for k in keys]) & (labels == 0) & sel
if ph.any():
    out["real_photos"] = {"n": int(ph.sum()), "pe_spc": float((fs[ph] < t_spc).mean()),
                          "ffaa": float((base[ph] < t_ffa).mean())}
    print(f"     {'photo_reals':<16} n={int(ph.sum()):>7,} "
          f"PE-SPC={out['real_photos']['pe_spc']:.6f} FFAA={out['real_photos']['ffaa']:.6f}")
out["real_ids"] = ids

# Score-distribution saturation: WHY one model has a real-recall ceiling the other does not.
print(f"\n-- score saturation on this view (the real-false-positive ceiling) --")
sat = {}
for who, sc in (("PE-SPC", fs), ("FFAA", base)):
    r = sc[sel & (labels == 0)]
    mx = float(r.max())
    frac_at_max = float((r >= mx).mean())
    sat[who] = {"n_real": int(len(r)), "max_real_score": mx, "frac_real_at_max": frac_at_max,
                "p99_real": float(np.percentile(r, 99))}
    print(f"     {who:<8} reals at the maximum score: {frac_at_max*100:.4f}% (max={mx:.4f}, "
          f"p99={np.percentile(r,99):.4f})")
out["score_saturation"] = sat
print(f"     => a model with X% of REALS at the top of its scale cannot exceed (100-X)% real recall at")
print(f"        any threshold. FFAA: {sat['FFAA']['frac_real_at_max']*100:.3f}% of reals at 1.0000, the")
print(f"        same value as most PAD frames, so 99% real recall is unreachable for it on this view.")

json.dump(out, open(a.out, "w"), indent=1)
np.save(a.out.replace(".json", "_scores.npy"), fs)
print(f"\n[axon1] -> {a.out}")

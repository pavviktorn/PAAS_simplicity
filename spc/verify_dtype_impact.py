#!/usr/bin/env python
"""Does the extraction dtype move a REPORTED NUMBER? (gate G3, the version that matters)

The cosine between fp32 and bf16->fp16 features is 0.999471 mean / 0.996998 min on this encoder. That
is a real difference, and the honest question is not "is the cosine above some constant I picked" but
"would the metric I report change if the features had been computed the other way". So: take the
TRAINED head, score the same images from fp32-computed and from bf16->fp16-computed features, and
compare the numbers that actually get published.

Criteria, pre-specified in verify_parity.py's docstring before any of this was measured:
    |delta bin_auc| <= 0.0005   and   |delta fake_rec@real98| <= 0.002

The eval manifest is BLOCK-ORDERED BY CLASS, so the subsample is strided and asserted to be 3-class.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "perception_models"))
sys.path.insert(0, os.path.join(ROOT, "spc"))
import core.vision_encoder.pe as pe                    # noqa: E402
import core.vision_encoder.transforms as pt            # noqa: E402
import metrics as M                                    # noqa: E402
from evaluate import load_head                         # noqa: E402

CKPT = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"
D_AUC, D_FR = 0.0005, 0.002
# The head this gate runs on must be a CANDIDATE model, not any head. The quantity being measured is
# "how many samples does a small feature perturbation push across the threshold", and that is governed
# by the sample density near the threshold -- i.e. by the head's separation, not by the cache. Measured
# directly: a 2%-data 1-epoch anchor (bin_auc 0.806, eer 0.266) showed |d fake_rec| = 0.0063 at a fixed
# tau, because with an EER of 0.27 a large fraction of samples sit within a whisker of any boundary. A
# head in the regime this project reports (GSD 0.9913, SeLop 0.9880) keeps most samples far from it.
# Below MIN_AUC the number would describe the head's fragility and be reported as a property of the
# dtype, so no verdict is rendered. This is NOT a relaxation of D_AUC/D_FR -- those are unchanged.
MIN_AUC = 0.99

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default=os.path.join(ROOT, "runs/spc/dtype_impact.json"))
a = ap.parse_args()

recs = json.load(open(os.path.join(ROOT, "manifests/eval.json")))
step = max(1, len(recs) // a.n)
sub = recs[::step][:a.n]
labels = np.array([r["label"] for r in sub])
assert len(set(labels.tolist())) == 3, f"strided subsample is not 3-class: {np.bincount(labels)}"
print(f"[dtype] {len(sub)} strided images, classes={np.bincount(labels).tolist()}")

head, ck = load_head(a.ckpt, a.device)
print(f"[dtype] head={ck['cfg'].get('name')} params={head.n_trainable():,}")

model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True, checkpoint_path=CKPT)
model = model.to(a.device).eval()
tfm = pt.get_image_transform(model.image_size)
x = torch.stack([tfm(Image.open(r["image"]).convert("RGB")) for r in sub])
print(f"[dtype] transformed {tuple(x.shape)}")


def feats(dtype, store):
    m = model.to(dtype)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            f = m.encode_image(x[i:i + 32].to(a.device, dtype=dtype), normalize=True)
            out.append(f.float().cpu().numpy().astype(store))
    return np.concatenate(out)


t0 = time.time()
f32 = feats(torch.float32, np.float32)
f16 = feats(torch.bfloat16, np.float16)          # exactly what the cache stores
print(f"[dtype] both passes in {time.time()-t0:.0f}s")

# The DEPLOYED threshold, applied to BOTH feature paths. Self-fitting a threshold separately for each
# dtype (which is all this file used to do) re-calibrates away exactly the effect being measured: if
# bf16 shifts the score distribution, two independently fitted taus both land on 98% real recall and
# the comparison looks clean while a fixed, shipped tau would have moved. This is the test that
# corresponds to what actually happens in production.
tau_dev = (ck.get("tau") or {}).get("98")
res = {"tau_devA_applied": tau_dev}
if tau_dev is None:
    print("[dtype] NOTE the checkpoint carries no DEV-A tau; only the self-fitted comparison is possible")
else:
    fixed = {}
    for nm, f in (("fp32", f32), ("bf16_fp16", f16)):
        with torch.no_grad():
            p_ = head(torch.from_numpy(f.astype(np.float32)).to(a.device)).softmax(-1).cpu().numpy()
        fsc = 1.0 - p_[:, 0]
        real = fsc[labels == 0]; fake = fsc[labels != 0]
        fixed[nm] = {"real_rec": float((real < tau_dev).mean()),
                     "fake_rec": float((fake >= tau_dev).mean())}
        print(f"[dtype] at the FIXED DEV-A tau={tau_dev:.6f}: {nm:<10} "
              f"real_rec={fixed[nm]['real_rec']:.6f} fake_rec={fixed[nm]['fake_rec']:.6f}")
    res["fixed_tau"] = fixed
    res["d_real_rec_fixed_tau"] = abs(fixed["fp32"]["real_rec"] - fixed["bf16_fp16"]["real_rec"])
    res["d_fake_rec_fixed_tau"] = abs(fixed["fp32"]["fake_rec"] - fixed["bf16_fp16"]["fake_rec"])
    print(f"[dtype] at the fixed tau: |d real_rec|={res['d_real_rec_fixed_tau']:.6f} "
          f"|d fake_rec|={res['d_fake_rec_fixed_tau']:.6f} (bound {D_FR} each)")

for nm, f in (("fp32", f32), ("bf16_fp16", f16)):
    with torch.no_grad():
        z = head(torch.from_numpy(f.astype(np.float32)).to(a.device))
        p = z.softmax(-1).cpu().numpy()
    fs = 1.0 - p[:, 0]
    m = M.block(fs, labels, p)
    res[nm] = {k: v for k, v in m.items() if isinstance(v, float)}
    print(f"[dtype] {nm:<10} bin_auc={m['bin_auc']:.6f} ap={m['ap']:.6f} "
          f"fake_rec@real98={m['fake_rec@real98']:.6f} eer={m['eer']:.6f}")

d_auc = abs(res["fp32"]["bin_auc"] - res["bf16_fp16"]["bin_auc"])
d_fr = abs(res["fp32"]["fake_rec@real98"] - res["bf16_fp16"]["fake_rec@real98"])

# COMPETENCE CHECK -- see MIN_AUC. Must precede the verdict.
weak = min(res["fp32"]["bin_auc"], res["bf16_fp16"]["bin_auc"])
if weak < MIN_AUC:
    res["no_verdict_reason"] = f"head too weak (bin_auc {weak:.6f} < {MIN_AUC})"
    json.dump(res, open(a.out, "w"), indent=1)
    raise SystemExit(
        f"[dtype] NO VERDICT: this head reaches bin_auc {weak:.6f} on the subsample, below {MIN_AUC}. "
        f"At that separation the measurement is dominated by how many samples sit near the threshold, "
        f"so it would report the head's fragility as a property of the feature dtype. Deltas observed "
        f"anyway, for the record: |d bin_auc|={abs(res['fp32']['bin_auc']-res['bf16_fp16']['bin_auc']):.6f}, "
        f"|d fr@real98|={abs(res['fp32']['fake_rec@real98']-res['bf16_fp16']['fake_rec@real98']):.6f}"
        + (f", |d fake_rec@fixed tau|={res['d_fake_rec_fixed_tau']:.6f}"
           if "d_fake_rec_fixed_tau" in res else "")
        + f". Re-run against a trained candidate head (stage2 uses runs/spc/spc_paper.pt on the full "
          f"trainset).")

# POWER CHECK. fake_rec@real98 is a fraction over the fake subset, so it can only move in steps of
# 1/n_fake; if that step is comparable to the bound, the test cannot distinguish a dtype effect from a
# single sample crossing the threshold. Observed directly: at n=512 (~340 fakes) the step is 0.0029 and
# the measured delta was 0.002882 -- i.e. exactly one sample, reported as a FAIL against a 0.002 bound.
# Refuse to render a verdict rather than emit a meaningless one.
n_fake = int((labels != 0).sum())
step = 1.0 / max(n_fake, 1)
res["n_fake"] = n_fake
res["fr_quantum"] = step
if step > D_FR / 2:
    need = int(np.ceil(2.0 / D_FR / max((labels != 0).mean(), 1e-9)))
    raise SystemExit(
        f"[dtype] UNDERPOWERED, no verdict: fake_rec@real98 moves in steps of 1/{n_fake} = {step:.6f}, "
        f"which is coarser than half the bound ({D_FR}). A single sample crossing tau would read as a "
        f"failure. Re-run with --n >= {need:,} (measured deltas so far: |d auc|={d_auc:.6f}, "
        f"|d fr|={d_fr:.6f}).")

ok = d_auc <= D_AUC and d_fr <= D_FR
# the fixed-threshold deltas are part of the gate, not decoration
for _k in ("d_real_rec_fixed_tau", "d_fake_rec_fixed_tau"):
    if _k in res and res[_k] > D_FR:
        print(f"[dtype] {_k}={res[_k]:.6f} exceeds {D_FR} -> the SHIPPED threshold moves with the dtype")
        ok = False
print(f"[dtype] |d bin_auc|={d_auc:.6f} (bound {D_AUC}) | |d fake_rec@real98|={d_fr:.6f} "
      f"(bound {D_FR}) -> {'PASS' if ok else 'FAIL'}")
res.update({"d_bin_auc": d_auc, "d_fake_rec_at_real98": d_fr, "bound_auc": D_AUC,
            "bound_fr": D_FR, "pass": bool(ok), "n": len(sub)})
json.dump(res, open(a.out, "w"), indent=1)
print(f"[dtype] -> {a.out}")
if not ok:
    raise SystemExit("[dtype] FATAL: the extraction dtype moves a reported metric beyond the "
                     "pre-specified bound -> re-extract the cache in fp32 (about 2 h on 4 GPUs).")

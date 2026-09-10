#!/usr/bin/env python
"""Single-image / directory inference with PE-SPC, with the fingerprint guards live.

The score served here is EXACTLY what evaluate.py measures:
    fake_score = 1 - softmax(logits)[REAL]
There is one preprocessing constructor (pt.get_image_transform) and the checkpoint records the
transform repr, image size, dtype and the encoder checkpoint sha256; a mismatch is a hard exit, not
a warning. PE-SPC must never be fed paas/preprocess.py letterbox output -- that is a different
distribution and it fails silently with a plausible-looking score.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "perception_models"))
sys.path.insert(0, os.path.join(ROOT, "spc"))
import core.vision_encoder.pe as pe                    # noqa: E402
import core.vision_encoder.transforms as pt            # noqa: E402
from head import build_head                            # noqa: E402

CKPT = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"
CLS = ("real", "pad", "deepfake")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--dir", default=None)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--threshold", type=float, default=None, help="default: tau@real98 from the ckpt")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--allow-dtype-mismatch", action="store_true",
                help="serve in a dtype other than the one the head was calibrated on. Off by default: "
                     "the shift is small but not nil and it moves a fixed threshold.")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True, checkpoint_path=CKPT)
model = model.to(a.device).eval().to(dt)
tfm = pt.get_image_transform(model.image_size)

fp = ck.get("fingerprint")
if fp:
    # The previous version built `live` with encoder_sha256 copied FROM the stored fingerprint, so the
    # comparison was tautological and only transform + image_size were ever checked. The encoder is the
    # single most important thing to verify -- swapping the weights changes every feature while the
    # transform and image size stay identical -- so it is hashed here for real.
    import hashlib as _h
    _sha = _h.sha256()
    with open(CKPT, "rb") as _f:
        for _c in iter(lambda: _f.read(1 << 22), b""):
            _sha.update(_c)
    live = {"transform": repr(tfm), "image_size": int(model.image_size),
            "encoder_sha256": _sha.hexdigest()}
    for k in ("transform", "image_size", "encoder_sha256"):
        if fp.get(k) is not None and fp[k] != live[k]:
            raise SystemExit(f"[infer] FINGERPRINT MISMATCH on {k}:\n  ckpt={fp[k]!r}\n  live={live[k]!r}\n"
                             f"[infer] the head was calibrated against different pixels or different "
                             f"encoder weights -- refusing to score.")
    if fp.get("dtype") and fp["dtype"] != a.dtype:
        if not a.allow_dtype_mismatch:
            raise SystemExit(
                f"[infer] DTYPE MISMATCH: serving in {a.dtype} but the head was calibrated on features "
                f"computed in {fp['dtype']}. The measured effect is not nil -- fp32 vs bf16 features "
                f"differ at cosine 0.9995 mean / 0.9970 min on this encoder, which is enough to move a "
                f"fixed threshold (see runs/spc/dtype_impact.json). Serve in {fp['dtype']}, or pass "
                f"--allow-dtype-mismatch to accept the shift deliberately.")
        print(f"[infer] WARNING serving in {a.dtype} against a head calibrated on {fp['dtype']} "
              f"features -- accepted via --allow-dtype-mismatch.")
    print(f"[infer] fingerprint OK: image_size={fp['image_size']}, encoder_sha256="
          f"{live['encoder_sha256'][:16]}..., dtype_trained={fp.get('dtype')}")
else:
    print("[infer] WARNING: checkpoint carries no fingerprint; cannot prove preprocessing parity")

head = build_head(ck["cfg"]).to(a.device); head.load_state_dict(ck["state_dict"]); head.eval()
tau = a.threshold if a.threshold is not None else ck.get("tau", {}).get("98")
print(f"[infer] cfg={ck['cfg'].get('name')} head={ck['cfg'].get('head')} "
      f"prompts={ck['cfg'].get('prompts')} params={head.n_trainable():,} tau={tau}")

paths = list(a.image)
if a.dir:
    for e in ("jpg", "jpeg", "png", "bmp", "webp"):
        paths += glob.glob(os.path.join(a.dir, "**", f"*.{e}"), recursive=True)
paths = sorted(set(paths))
if not paths:
    raise SystemExit("[infer] no images")

t0 = time.time()
with torch.no_grad():
    for i in range(0, len(paths), a.batch_size):
        chunk = paths[i:i + a.batch_size]
        x = torch.stack([tfm(Image.open(p).convert("RGB")) for p in chunk]).to(a.device, dtype=dt)
        p = head(model.encode_image(x, normalize=True).float()).softmax(-1).cpu().numpy()
        for pth, pr in zip(chunk, p):
            fake = 1.0 - pr[0]
            verdict = "-" if tau is None else ("FAKE" if fake >= tau else "REAL")
            print(f"{verdict:<5} fake={fake:.6f} p_real={pr[0]:.4f} p_pad={pr[1]:.4f} "
                  f"p_deepfake={pr[2]:.4f} argmax={CLS[int(pr.argmax())]:<9} {pth}")
print(f"[infer] {len(paths)} images in {time.time()-t0:.1f}s", flush=True)

#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Extract FROZEN Perception Encoder features once, cache to disk.

PE-SPC trains only C x 1280 prototype values, so the encoder forward is ~100% of the cost and is
completely independent of the prompt choice and of every hyperparameter. Extracting once and
training on cached features turns a 2.5-hour experiment into a 2-second one, which is what makes an
honest prompt search affordable (dozens of candidate triplets, each evaluated on real data).

Preprocessing: PE's OWN transform (Resize(448,448) bilinear+antialias, Normalize(0.5,0.5)).
  * That is a SQUASH resize: it keeps the WHOLE FRAME (no crop). Both papers say "resized and
    center-cropped"; a center crop would throw away the frame borders where presentation-attack
    evidence lives (screen edges, bezels, moire) -- the deviation is deliberate and recorded.
  * It is also exactly how the encoder was pretrained, so there is no train/serve preprocessing
    skew and no letterbox gray bars outside the pretraining distribution.

  VENV=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python  (PYTHONNOUSERSITE=1)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PM = "/datasets/work/vLLM/temp/PAAS_simplicity/perception_models"
sys.path.insert(0, PM)
sys.path.insert(0, "/datasets/work/vLLM/temp/PAAS_ensemble_v4/gsd")
import core.vision_encoder.pe as pe                      # noqa: E402
import core.vision_encoder.transforms as pt              # noqa: E402
from get_label import get_label_all, MAKEUP, PAD, UNKNOWN  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance as PROV                              # noqa: E402

CKPT = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"

# ---- unreadable-image guard -------------------------------------------------------------------
# A silent black-frame substitution corrupted a GSD anchor earlier in this project: 32% of an eval
# set became constant images and every log line still looked healthy. Never substitute silently.
_MISS = {"n": 0, "seen": 0}
_MISS_MAX = int(os.environ.get("IMG_MISS_MAX", "100"))
_MISS_RATE = float(os.environ.get("IMG_MISS_RATE", "0.01"))


class ImageList(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        _MISS["seen"] += 1
        try:
            img = Image.open(path).convert("RGB")
            ok = 1
        except Exception as exc:
            _MISS["n"] += 1
            n, seen = _MISS["n"], _MISS["seen"]
            if n <= 10:
                print(f"[extract] UNREADABLE #{n}: {path} ({type(exc).__name__})", flush=True)
            if n > _MISS_MAX or (seen >= 200 and n / seen > _MISS_RATE):
                raise RuntimeError(
                    f"[extract] too many unreadable images: {n}/{seen} ({n/seen*100:.2f}%). "
                    f"Features for these would be a constant vector and would silently poison "
                    f"both the prototypes and every metric. Fix the data.") from exc
            img = Image.new("RGB", (448, 448))
            ok = 0
        return self.tfm(img), int(label), i, ok


def build_items(src, limit=0):
    """(path, 3-class label) list. MAKEUP -> PAD, UNKNOWN dropped. Labels from the PATH."""
    if src.endswith(".json"):
        recs = json.load(open(src))
        paths = [r.get("image") or r.get("path") for r in recs]
    else:
        paths = []
        for dp, _, fs in os.walk(src):
            paths += [os.path.join(dp, f) for f in sorted(fs)
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        paths.sort()
    items, dropped = [], 0
    for p in paths:
        lab = get_label_all(p)
        if lab == UNKNOWN:
            dropped += 1
            continue
        items.append((p, PAD if lab == MAKEUP else lab))
        if limit and len(items) >= limit:
            break
    return items, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="json manifest or image directory")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>_<part>.npz)")
    ap.add_argument("--which-part", type=int, default=0)
    ap.add_argument("--n-divided", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out = f"{a.out}_{a.which_part}.npz"

    items, dropped = build_items(a.src, a.limit)
    mine = items[a.which_part::a.n_divided]           # stride, same convention as the v4 pipeline
    print(f"[extract] shard {a.which_part}/{a.n_divided}: {len(mine):,} of {len(items):,} "
          f"(dropped {dropped:,} UNKNOWN)", flush=True)

    model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True, checkpoint_path=CKPT)
    model = model.cuda().eval().to(torch.bfloat16)
    tfm = pt.get_image_transform(model.image_size)

    # Provenance BEFORE the skip decision: "the file exists" is not a reason to reuse features. If the
    # manifest, the encoder, the transform or the shard count changed, the cached rows describe
    # different images and reusing them mis-aligns every label.
    meta = PROV.build(a.src, CKPT, repr(tfm), model.image_size, "bf16", "fp16",
                      a.n_divided, a.which_part, len(items))
    if os.path.exists(out):
        ok, why = PROV.check_reusable(out, meta)
        print(f"[extract] {out} exists -> {'skip' if ok else 'REFUSE'} ({why})", flush=True)
        if ok:
            return                      # never stamp a shard we only reused -- see check_reusable
        raise SystemExit(f"[extract] refusing to reuse {out}: {why}. Delete it and re-extract.")

    dl = DataLoader(ImageList(mine, tfm), batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, pin_memory=True, prefetch_factor=4)

    F_, L_, I_, OK_ = [], [], [], []
    t0, done = time.time(), 0
    with torch.no_grad():
        for x, y, idx, ok in dl:
            f = model.encode_image(x.cuda(non_blocking=True).to(torch.bfloat16), normalize=True)
            F_.append(f.float().cpu().numpy().astype(np.float16))
            L_.append(y.numpy().astype(np.int8)); I_.append(idx.numpy().astype(np.int64))
            OK_.append(ok.numpy().astype(np.int8))
            done += x.shape[0]
            if done % (a.batch_size * 40) == 0:
                el = time.time() - t0
                print(f"[extract] shard {a.which_part}: {done:,}/{len(mine):,} "
                      f"{done/el:.1f} img/s eta {(len(mine)-done)/max(done/el,1e-9)/60:.0f}m",
                      flush=True)
    feats = np.concatenate(F_); labs = np.concatenate(L_)
    okv = np.concatenate(OK_)
    # NOTE: _MISS["n"] lives in the PARENT, but __getitem__ runs in FORKED WORKERS, so the parent's
    # counter never moves and `unreadable=_MISS["n"]` was structurally always 0 -- a log line that
    # could only ever say "healthy". The per-item `ok` flag is returned THROUGH the DataLoader, so it
    # is the only trustworthy source. Derive the count from it. (Same reason the abort threshold in
    # __getitem__ is per-worker and thus ~workers x looser than IMG_MISS_MAX suggests; verify_cache.py
    # is the real gate.)
    n_bad = int((okv == 0).sum())
    np.savez(out, feats=feats, labels=labs, index=np.concatenate(I_), ok=okv,
             paths=np.array([p for p, _ in mine], dtype=object), unreadable=n_bad)
    PROV.write(out, meta)
    print(f"[extract] shard {a.which_part} DONE {feats.shape} unreadable={n_bad} (from ok[]) "
          f"{time.time()-t0:.0f}s -> {out} (+ provenance sidecar)", flush=True)


if __name__ == "__main__":
    main()

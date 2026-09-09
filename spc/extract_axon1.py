"""Extract PE features for the axon1 held-out set (614,029 frames from 2,350 media files).

axon1 frames are NOT files on disk -- they are video frames addressed as `<path>#frame=NNNNNN`. The
frames are decoded with cv2 in sequential order and keyed exactly as scripts/run_dataset.py keys them
(`f"{path}#frame={i:06d}"`), so every feature joins to the recorded baseline score BY KEY. That
removes the ~3 pp frame-alignment noise the previous axon1 comparisons had to carry as a caveat:
same frame, same key, or it is not counted.

Sharding is by MEDIA FILE, never mid-file (the project convention), so every frame of a video stays
in one shard and one decode pass serves it.

Labels come from the axon1 directory layout: `fake/pad/*` -> PAD(1), `real/*` -> REAL(0). axon1
contains NO deepfake class -- it is a PAD-vs-real benchmark, and the report must say so rather than
quoting a 3-class number that has an empty cell.
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time

# Thread hygiene: this script runs 2 processes per GPU and cv2's decoder plus torch's CPU ops will
# each grab every core by default, so 8 shards would spend their time context-switching instead of
# decoding. Measured cause of a starved throughput test, not a guess.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("AXON1_THREADS", "3"))
import numpy as np
import torch
from PIL import Image

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "perception_models"))
import core.vision_encoder.pe as pe                  # noqa: E402
import core.vision_encoder.transforms as pt          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance as PROV                              # noqa: E402

CKPT = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"
torch.set_num_threads(int(os.environ.get("AXON1_THREADS", "3")))
try:
    import cv2
    cv2.setNumThreads(int(os.environ.get("AXON1_CV_THREADS", "2")))
except Exception:
    pass
AXON1_RESULTS = ("/datasets/work/vLLM/temp/PAAS_ensemble_v2/runs/"
                 "test_axon0model_axon1datatest/results_ffaa.txt")
PRE = "/datasets/work/vLLM/data/axonlabs_data_1/"


def parse_results(path=AXON1_RESULTS):
    """-> ordered dict media_path -> list[(frame_idx, key, baseline_fake_score)], plus label map."""
    per = collections.OrderedDict()
    labels = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.startswith("OK"):
            continue                                  # SK (lowqual) / ER were not scored -> skip
        i = ln.index(PRE)
        rest = ln[i:].rstrip("\n")
        # the baseline score sits in the `fake=` column; paths contain spaces so split on the marker
        head = ln[:i]
        fake = float(head.split("fake=")[1].split()[0])
        if "#frame=" in rest:
            media, fr = rest.split("#frame=")
            fidx = int(fr)
        else:
            media, fidx = rest, -1
        rel = media[len(PRE):]
        labels[media] = 0 if rel.startswith("real/") else 1        # axon1: real(0) vs pad(1)
        per.setdefault(media, []).append((fidx, rest, fake))
    return per, labels


def frames_of(media, wanted):
    """Yield (key, PIL.Image) for the wanted frame indices, decoding sequentially like
    scripts/run_dataset.py. `wanted` maps frame_idx -> key."""
    if media.lower().endswith((".jpg", ".jpeg", ".png")):
        yield wanted[-1], Image.open(media).convert("RGB")
        return
    import cv2
    cap = cv2.VideoCapture(media)
    i, need = 0, len(wanted)
    got = 0
    while got < need:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            yield wanted[i], Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            got += 1
        i += 1
    cap.release()
    if got < need:
        print(f"[axon1] WARNING decoded {got}/{need} wanted frames from {media} "
              f"(video shorter than the baseline run saw)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "cache/axon1"))
    ap.add_argument("--which-part", type=int, default=0)
    ap.add_argument("--n-divided", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-media", type=int, default=0)
    a = ap.parse_args()
    out = f"{a.out}_{a.which_part}.npz"
    per, labmap = parse_results()
    media = list(per)
    mine = media[a.which_part::a.n_divided]
    if a.limit_media:
        mine = mine[:a.limit_media]
    nfr = sum(len(per[m]) for m in mine)
    print(f"[axon1] shard {a.which_part}/{a.n_divided}: {len(mine):,} media / {nfr:,} frames "
          f"(of {len(media):,} media / {sum(len(v) for v in per.values()):,} frames)", flush=True)

    model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True, checkpoint_path=CKPT)
    model = model.cuda().eval().to(torch.bfloat16)
    tfm = pt.get_image_transform(model.image_size)

    meta = PROV.build(AXON1_RESULTS, CKPT, repr(tfm), model.image_size, "bf16", "fp16",
                      a.n_divided, a.which_part, len(media))
    if os.path.exists(out):
        ok, why = PROV.check_reusable(out, meta)
        print(f"[axon1] {out} exists -> {'skip' if ok else 'REFUSE'} ({why})", flush=True)
        if ok:
            return                      # never stamp a shard we only reused -- see check_reusable
        raise SystemExit(f"[axon1] refusing to reuse {out}: {why}. Delete it and re-extract.")

    F_, K_, L_, B_ = [], [], [], []
    buf, bkey = [], []
    t0, done, bad = time.time(), 0, 0

    def flush():
        nonlocal buf, bkey
        if not buf:
            return
        x = torch.stack(buf).cuda(non_blocking=True).to(torch.bfloat16)
        with torch.no_grad():
            f = model.encode_image(x, normalize=True)
        F_.append(f.float().cpu().numpy().astype(np.float16))
        K_.extend(bkey)
        buf, bkey = [], []

    for mi, m in enumerate(mine):
        rows = per[m]
        wanted = {fidx: key for fidx, key, _ in rows}
        if -1 in wanted:                                     # standalone photo
            wanted = {-1: rows[0][1]}
        score = {key: s for _, key, s in rows}
        lab = labmap[m]
        try:
            for key, img in frames_of(m, wanted):
                buf.append(tfm(img)); bkey.append(key)
                L_.append(lab); B_.append(score[key]); done += 1
                if len(buf) >= a.batch_size:
                    flush()
        except Exception as exc:
            bad += 1
            print(f"[axon1] DECODE FAILED ({type(exc).__name__}) {m}", flush=True)
        if (mi + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[axon1] shard {a.which_part}: media {mi+1:,}/{len(mine):,} frames {done:,}/{nfr:,} "
                  f"{done/el:.1f} img/s eta {(nfr-done)/max(done/el,1e-9)/60:.0f}m", flush=True)
    flush()
    feats = np.concatenate(F_) if F_ else np.zeros((0, 1280), np.float16)
    np.savez(out, feats=feats, labels=np.array(L_, np.int8), keys=np.array(K_, dtype=object),
             baseline=np.array(B_, np.float32), ok=np.ones(len(L_), np.int8),
             n_media=len(mine), n_decode_failed=bad)
    PROV.write(out, meta)
    print(f"[axon1] shard {a.which_part} DONE {feats.shape} decode_failed_media={bad} "
          f"{time.time()-t0:.0f}s -> {out} (+ provenance sidecar)", flush=True)


if __name__ == "__main__":
    main()

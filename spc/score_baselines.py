#!/usr/bin/env python
"""Score the PREVIOUS models on the SAME testset images PE-SPC is evaluated on, in one pass, so the
comparison table is measured under one harness instead of quoted across harnesses.

Members scored here (all MLLM-free and cheap): mids9c/A2, GSD, SeLop -- via the standalone
PAAS_ensemble_v4_inf pipeline, which exposes per-member scores (`ensemble_fake`, `gsd_fake`,
`selop_fake`).

mids9c is re-scored ON PURPOSE: its own training log reports val acc 0.9995 / auc 1.0000 / ap 1.0000,
which is a saturated (and probably leaked) split and cannot be used as a baseline. The score used
here is the ensemble's MARGINAL 9-class score, exactly as production computes it
(`Pt = softmax.view(n,3,3).sum(2); fake = mean_a(1 - Pt[:,0])`), never the selector score.

FFAA (Qwen3.5-4B + from-scratch MIDS) is NOT re-run: it needs the 4B MLLM at ~9.3 fps (~54 min) and
it already has a recorded number on this exact 30,197-image testset. That recorded number is quoted
with its provenance rather than silently re-measured under different conditions.
"""
import argparse, json, os, sys, time
import numpy as np

V4 = "/datasets/work/vLLM/temp/PAAS_ensemble_v4_inf"
ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "spc"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifests/eval.json"))
    # paas_seed1981723.json, NOT paas4_qwen.json. The deployed weights ARE the seed-1981723 build, and
    # only that config sets ffaa.whole_frame=true. paas4_qwen.json omits it, so it defaults to False and
    # the MIDS head would be fed a CLIP CENTER CROP while it was trained on whole letterboxed frames --
    # a train/serve skew inside the BASELINE, which would have understated the strongest model I am
    # comparing against, in my own favour. (SeLop is safe either way: it reads whole_frame from inside its
    # own checkpoint. GSD and the 9-class members carry their own preprocessing too. FFAA's MIDS head is
    # the only one that takes it from the config file.)
    ap.add_argument("--config", default=os.path.join(V4, "config/experiments/paas_seed1981723.json"))
    ap.add_argument("--allow-crop-mids", action="store_true",
                    help="permit ffaa.whole_frame=false. Off by default: the deployed MIDS head was "
                         "trained on whole frames.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    # Shard the testset across GPUs. FFAA is the slow member (~3 img/s even at bs128 while the axon1
    # decode contends for CPU), so a single process needs ~2.8 h for 30,197 images. Four shards on four
    # GPUs turn that into ~30 min and let the comparison use the FULL testset instead of a subsample.
    ap.add_argument("--which-part", type=int, default=0)
    ap.add_argument("--n-divided", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/spc/baselines_testset.json"))
    ap.add_argument("--with-ffaa", action="store_true",
                    help="also score FFAA (Qwen3.5-4B + MIDS). ~9.3 fps, so ~54 min for 30,197 images. "
                         "Worth it because the recorded FFAA number has no per-class pad/deepfake recall, "
                         "and it puts the strongest baseline in the SAME harness as everything else.")
    a = ap.parse_args()

    sys.path.insert(0, V4)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device.split(":")[-1])
    from paas import env as penv                                    # noqa: E402
    penv.setup(device="cuda:0")
    from paas.config import PaasConfig                              # noqa: E402
    from paas.pipeline import PaasPipeline as Pipeline                              # noqa: E402
    from PIL import Image                                           # noqa: E402

    # FFAA off: it is the expensive member (~9.3 fps) and it already has a recorded number on this exact
    # testset. PaasConfig.validate() rejects a disabled member that fusion still lists, so drop it from
    # `components` too -- the fused score is not used here, only the per-member ones.
    d = json.load(open(a.config))
    if not a.with_ffaa:
        d["ffaa"]["enabled"] = False
        d["fusion"]["components"] = [c for c in d["fusion"]["components"] if c != "ffaa"]
    cfg = PaasConfig.from_dict(d).validate()
    wf = bool(getattr(cfg.ffaa, "whole_frame", False))
    print(f"[base] config={os.path.basename(a.config)} members={cfg.fusion.components} "
          f"ffaa.whole_frame={wf} ffaa.enabled={cfg.ffaa.enabled}", flush=True)
    if a.with_ffaa and not wf and not a.allow_crop_mids:
        raise SystemExit(
            "[base] ffaa.whole_frame=False but the deployed MIDS head was trained on WHOLE FRAMES. "
            "Scoring it with a CLIP centre crop is a train/serve skew that silently weakens the baseline. "
            "Use config/experiments/paas_seed1981723.json, or pass --allow-crop-mids deliberately.")
    pipe = Pipeline(cfg)

    recs = json.load(open(a.manifest))
    if a.limit:
        recs = recs[:: max(1, len(recs) // a.limit)][:a.limit]
    if a.n_divided > 1:
        # stride, not block: eval.json is ordered by class, so contiguous slices would be single-class
        recs = recs[a.which_part::a.n_divided]
        print(f"[base] shard {a.which_part}/{a.n_divided}: {len(recs):,} images", flush=True)
    print(f"[base] {len(recs):,} images | config={os.path.basename(a.config)}", flush=True)

    MEMBERS = ["ensemble_fake", "gsd_fake", "selop_fake"] + (["ffaa_fake"] if a.with_ffaa else [])
    out = {"image": [], "label": []}
    for _k in MEMBERS:
        out[_k] = []
    t0 = time.time()
    for i in range(0, len(recs), a.batch_size):
        chunk = recs[i:i + a.batch_size]
        rgb, keep = [], []
        for r in chunk:
            try:
                rgb.append(np.array(Image.open(r["image"]).convert("RGB")))
                keep.append(r)
            except Exception as exc:
                print(f"[base] UNREADABLE {r['image']} ({type(exc).__name__})", flush=True)
        if not rgb:
            continue
        res = pipe.predict_frames(rgb, keys=[r["image"] for r in keep])
        for r, d in zip(keep, res):
            out["image"].append(r["image"]); out["label"].append(int(r["label"]))
            for k in MEMBERS:
                v = d.get(k)
                out[k].append(float(v) if v is not None else float("nan"))
        if (i // a.batch_size) % 20 == 0:
            el = time.time() - t0
            n = len(out["image"])
            print(f"[base] {n:,}/{len(recs):,} {n/el:.1f} img/s eta {(len(recs)-n)/max(n/el,1e-9)/60:.0f}m",
                  flush=True)

    import metrics as M                                             # noqa: E402
    lab = np.array(out["label"])
    summary = {}
    for k in MEMBERS:
        s = np.array(out[k], dtype=np.float64)
        ok = np.isfinite(s)
        m = M.block(s[ok], lab[ok])
        summary[k] = m
        print(f"\n=== {k} (n={int(ok.sum()):,}) ===\n  {M.fmt(m)}", flush=True)
        for t in (95, 98, 99):
            print(f"  tau@real{t}={m[f'tau@real{t}']:.6f} achieved_real={m[f'real_rec@real{t}']:.6f} "
                  f"fake_rec={m[f'fake_rec@real{t}']:.6f} (pad {m[f'pad_rec@real{t}']:.6f} / "
                  f"deepfake {m[f'deepfake_rec@real{t}']:.6f})", flush=True)
    json.dump({"n": len(out["image"]), "summary": summary, "members": MEMBERS,
               "scores": {k: out[k] for k in MEMBERS},
               "label": out["label"], "image": out["image"]}, open(a.out, "w"), indent=1)
    print(f"\n[base] -> {a.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    # REQUIRED, not decoration. vLLM forces the `spawn` start method once CUDA is initialised, and
    # spawn RE-IMPORTS this module in the child. As a flat top-level script it therefore re-ran itself
    # and tried to build a second vLLM engine, which died in multiprocessing.spawn.prepare() with
    # "Engine core initialization failed". A main guard is what makes the child import a no-op.
    main()

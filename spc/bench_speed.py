"""Speed benchmark for PE-SPC on ONE GPU.

Protocol chosen so the number means something and can be compared to the recorded FFAA/GSD/SeLop
figures instead of being quoted next to them:
  * exclusive GPU (the 47 img/s in the extraction logs is a 4-way-contended, 12-worker,
    JPEG-decode-included number and is NOT the encoder's speed -- quoting it would be the same
    class of error as quoting a leaked AUC);
  * 20 warmup batches discarded, then 200 timed batches, cuda.synchronize() around each region;
  * TWO numbers, always separate: GPU-only (tensor already on device) and end-to-end
    (Image.open -> transform -> encode -> head -> decision), which is what a service delivers.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "perception_models"))
import core.vision_encoder.pe as pe                  # noqa: E402
import core.vision_encoder.transforms as pt          # noqa: E402
from head import build_head                          # noqa: E402

CKPT = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"


def timed(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lat = []
    t0 = time.perf_counter()
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    return total, np.array(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained SPC head (optional; a fresh head is fine "
                                                "for timing -- 3,843 params cost nothing)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batches", default="1,8,32,64,128")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--images", default=os.path.join(ROOT, "manifests/eval.json"))
    ap.add_argument("--e2e-n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/spc/bench.json"))
    a = ap.parse_args()

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    torch.cuda.set_device(a.device)
    t0 = time.time()
    model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True, checkpoint_path=CKPT)
    model = model.to(a.device).eval().to(dt)
    load_s = time.time() - t0
    tfm = pt.get_image_transform(model.image_size)
    vis_params = sum(p.numel() for p in model.visual.parameters())
    all_params = sum(p.numel() for p in model.parameters())

    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
        head = build_head(ck["cfg"]).to(a.device); head.load_state_dict(ck["state_dict"])
    else:
        head = build_head({"head": "H1", "learn_scale": True}).to(a.device)
    head.eval()

    res = {"device": torch.cuda.get_device_name(a.device), "dtype": a.dtype,
           "image_size": int(model.image_size), "load_s": round(load_s, 1),
           "params_frozen_total": all_params, "params_frozen_vision": vis_params,
           "params_trained": head.n_trainable(), "iters": a.iters, "gpu_only": {}, "e2e": {}}
    print(f"[bench] {res['device']} dtype={a.dtype} load={load_s:.1f}s "
          f"frozen={all_params/1e9:.3f}B (vision {vis_params/1e9:.3f}B) trained={head.n_trainable():,}",
          flush=True)

    # ---- (a) GPU-only: tensor already on device -------------------------------------------------
    with torch.no_grad():
        for bs in [int(x) for x in a.batches.split(",")]:
            x = torch.randn(bs, 3, model.image_size, model.image_size, device=a.device, dtype=dt)

            def step():
                f = model.encode_image(x, normalize=True)
                return head(f.float()).softmax(-1)

            iters = a.iters if bs <= 32 else max(50, a.iters // 4)
            total, lat = timed(step, 20, iters)
            ips = bs * iters / total
            res["gpu_only"][bs] = {"img_s": round(ips, 2), "iters": iters,
                                   "lat_mean_ms": round(lat.mean() * 1e3, 3),
                                   "lat_p50_ms": round(float(np.percentile(lat, 50)) * 1e3, 3),
                                   "lat_p99_ms": round(float(np.percentile(lat, 99)) * 1e3, 3),
                                   "peak_mem_gb": round(torch.cuda.max_memory_allocated(a.device) / 2**30, 2)}
            print(f"[bench] GPU-only bs={bs:<4} {ips:8.2f} img/s  p50 {res['gpu_only'][bs]['lat_p50_ms']:8.2f}ms "
                  f"p99 {res['gpu_only'][bs]['lat_p99_ms']:8.2f}ms  peak {res['gpu_only'][bs]['peak_mem_gb']}GB",
                  flush=True)

    # ---- (b) end-to-end: disk -> PIL -> transform -> encode -> head -> decision -----------------
    recs = json.load(open(a.images))
    paths = [r["image"] for r in recs][:: max(1, len(recs) // a.e2e_n)][:a.e2e_n]
    paths = [p for p in paths if os.path.exists(p)]
    print(f"[bench] end-to-end over {len(paths)} real images (strided over the manifest, which is "
          f"block-ordered by class)", flush=True)
    with torch.no_grad():
        for bs in (1, 8, 32):
            lat, n = [], 0
            for _ in range(3 if bs > 1 else 1):                      # warm the page cache
                for i in range(0, len(paths) - bs + 1, bs):
                    s = time.perf_counter()
                    batch = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
                    f = model.encode_image(batch.to(a.device, dtype=dt), normalize=True)
                    _ = head(f.float()).softmax(-1)[:, 0]
                    torch.cuda.synchronize()
                    lat.append(time.perf_counter() - s); n += bs
            l = np.array(lat[len(lat) // 3:])                        # drop the cold third
            res["e2e"][bs] = {"img_s": round(bs / l.mean(), 2), "n": n,
                              "lat_mean_ms": round(l.mean() * 1e3, 3),
                              "lat_p50_ms": round(float(np.percentile(l, 50)) * 1e3, 3),
                              "lat_p99_ms": round(float(np.percentile(l, 99)) * 1e3, 3)}
            print(f"[bench] e2e      bs={bs:<4} {res['e2e'][bs]['img_s']:8.2f} img/s  "
                  f"p50 {res['e2e'][bs]['lat_p50_ms']:8.2f}ms p99 {res['e2e'][bs]['lat_p99_ms']:8.2f}ms",
                  flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[bench] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

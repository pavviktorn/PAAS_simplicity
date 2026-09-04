#!/usr/bin/env python
"""Record FFAA's measured throughput from the sharded baseline run, per GPU and in aggregate.

The recorded project figure is 9.27 fps. The 4-way sharded run lands on ~9.2 img/s in AGGREGATE,
which suggests the recorded number is a multi-GPU total. Quoting it against a single-GPU PE-SPC
measurement would overstate FFAA roughly 4x -- the same class of error as quoting a recorded accuracy
whose conditions were never established.
"""
import glob, json, os, re, sys
R = "/datasets/work/vLLM/temp/PAAS_simplicity/runs/spc"
per = []
for f in sorted(glob.glob(os.path.join(R, "baselines_shard*.log"))):
    txt = open(f, errors="ignore").read()
    m = re.findall(r"->\s+\S+baselines_shard\d+\.json\s+\((\d+)s\)", txt)
    n = re.findall(r"shard \d+/\d+:\s+([\d,]+) images", txt)
    if m and n:
        secs = int(m[-1]); imgs = int(n[-1].replace(",", ""))
        per.append({"log": os.path.basename(f), "images": imgs, "seconds": secs,
                    "img_s": round(imgs / secs, 3)})
if not per:
    print("[ffaa] shard logs do not yet carry a completion line; run this after step 4 finishes")
    sys.exit(0)
agg = sum(p["images"] for p in per) / max(p["seconds"] for p in per)
out = {"per_shard": per, "n_gpus": len(per),
       "per_gpu_img_s": round(sum(p["img_s"] for p in per) / len(per), 3),
       "aggregate_img_s": round(agg, 3),
       "recorded_project_figure_fps": 9.27,
       "note": ("aggregate is total images divided by the SLOWEST shard's wall time, i.e. the real "
                "end-to-end rate of the 4-GPU run; per_gpu is the mean of the individual shard rates")}
json.dump(out, open(os.path.join(R, "ffaa_throughput.json"), "w"), indent=1)
for p in per:
    print(f"  {p['log']}: {p['images']:,} imgs in {p['seconds']}s = {p['img_s']} img/s")
print(f"  per-GPU mean {out['per_gpu_img_s']} img/s | aggregate {out['aggregate_img_s']} img/s "
      f"over {out['n_gpus']} GPUs (recorded project figure: 9.27 fps)")
print(f"  -> {R}/ffaa_throughput.json")

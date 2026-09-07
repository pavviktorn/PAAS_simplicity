"""Build manifests/axon1.json + the recorded FFAA baseline scores, from the v2 results file that
qwen/gen_score_axon1.py wrote.

Labels come ONLY from get_label_all(path) (project rule G5: the stored `truth` column is binary and
disagrees with the path on PAD). The file's own truth/score columns are kept for cross-tabulation and
for the baseline comparison, never as the label source.
"""
import json, os, sys, collections
sys.path.insert(0, "/datasets/work/vLLM/temp/PAAS_ensemble_v4/gsd")
from get_label import get_label_all, MAKEUP, PAD, UNKNOWN

SRC = "/datasets/work/vLLM/temp/PAAS_ensemble_v2/runs/test_axon0model_axon1datatest/results_ffaa.txt"
ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"

rows, stat, xtab = [], collections.Counter(), collections.Counter()
miss = 0
for ln in open(SRC):
    if ln.startswith("#"):
        continue
    f = ln.rstrip("\n").split("\t")
    if len(f) < 7:
        f = ln.split()
        if len(f) < 7:
            continue
    st, truth, pred, typ, fake, match, img = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
    stat[st] += 1
    if st in ("SK", "ER"):
        continue
    lab = get_label_all(img)
    if lab == UNKNOWN:
        stat["UNKNOWN_path_label"] += 1
        continue
    lab = PAD if lab == MAKEUP else lab
    xtab[(truth, lab)] += 1
    if not os.path.exists(img):
        miss += 1
        continue
    try:
        fs = float(fake)
    except ValueError:
        fs = float("nan")
    rows.append({"image": img, "label": int(lab), "ffaa_fake_score": fs, "file_truth": truth})

print(f"[axon1] status counts: {dict(stat)}")
print(f"[axon1] rows with an existing file: {len(rows):,} (dropped {miss:,} non-existent paths)")
h = collections.Counter(r["label"] for r in rows)
print(f"[axon1] path-derived labels: real={h[0]:,} pad={h[1]:,} deepfake={h[2]:,}")
print(f"[axon1] cross-tab (file_truth -> path label): "
      + " ".join(f"{k[0]}->{k[1]}:{v:,}" for k, v in sorted(xtab.items())))
json.dump(rows, open(f"{ROOT}/manifests/axon1.json", "w"))
print(f"[axon1] -> {ROOT}/manifests/axon1.json")

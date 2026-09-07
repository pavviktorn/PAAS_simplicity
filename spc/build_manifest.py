#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Build a 3-class manifest [{image,label}] with labels from the PATH (get_label_all).

Why this is a separate, explicit step: mids_first_half.json is a STALE manifest -- it references
350,034 images under /datasets/work/vLLM/data/fmt_error_all, which is now an EMPTY directory.
Feeding it straight to the extractor would have turned 20.5% of the trainset into black frames.
Filtering here, once, makes the real trainset size an explicit recorded number instead of a silent
subset, and keeps the extractor honest (its guard then has nothing to tolerate).
"""
import argparse, collections, json, os, sys
sys.path.insert(0, "/datasets/work/vLLM/temp/PAAS_ensemble_v4/gsd")
from get_label import get_label_all, MAKEUP, PAD, UNKNOWN  # noqa: E402

NAMES = {0: "real", 1: "pad", 2: "deepfake"}
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-exist-check", action="store_true")
    a = ap.parse_args()

    if a.src.endswith(".json"):
        paths = [r.get("image") or r.get("path") for r in json.load(open(a.src))]
    else:
        paths = []
        for dp, _, fs in os.walk(a.src):
            paths += [os.path.join(dp, f) for f in sorted(fs) if f.lower().endswith(EXT)]
        paths.sort()

    items, n_unknown, n_gone = [], 0, collections.Counter()
    for p in paths:
        lab = get_label_all(p)
        if lab == UNKNOWN:
            n_unknown += 1
            continue
        if not a.no_exist_check and not os.path.exists(p):
            n_gone["/".join(p.split("/")[:6])] += 1
            continue
        items.append({"image": p, "label": int(PAD if lab == MAKEUP else lab)})

    hist = collections.Counter(i["label"] for i in items)
    tot = max(len(items), 1)
    json.dump(items, open(a.out, "w"))
    print(f"[manifest] {a.src}")
    print(f"[manifest]   input paths     : {len(paths):,}")
    print(f"[manifest]   dropped UNKNOWN : {n_unknown:,}")
    print(f"[manifest]   dropped MISSING : {sum(n_gone.values()):,}")
    for k, v in n_gone.most_common(4):
        print(f"[manifest]       {v:>9,}  {k}")
    print(f"[manifest]   KEPT            : {len(items):,}")
    for c in sorted(hist):
        print(f"[manifest]       {NAMES[c]:9} {hist[c]:>9,}  ({hist[c]/tot*100:5.2f}%)")
    print(f"[manifest]   -> {a.out}")


if __name__ == "__main__":
    main()

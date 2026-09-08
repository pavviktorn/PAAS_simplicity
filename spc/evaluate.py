"""Score a trained SPC head on a feature cache and print the FULL metric block.

Two rules this file exists to enforce:
  1. Thresholds are fitted on DEV-A (a group-disjoint split of the TRAINSET) and applied UNCHANGED
     to the testset / axon1, with the ACHIEVED real recall printed next to every target. A threshold
     fitted on the set it is reported on is not a result.
  2. Breakdowns, not one number. `bin_auc` hides everything that matters here: which of the 6 real
     identities fails (the documented real-FP wall is R_12/R_13/R_15) and which of the 7 PAD attack
     families fails. GSD reads 0.9913 AUC while its deepfake recall is 0.813.
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache_io, metrics as M                        # noqa: E402
from head import build_head                          # noqa: E402
import train_spc as T                                # noqa: E402

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
ATTACKS = ["Replay_PC", "Replay_mobile", "Textile", "Advanced", "Silicone", "Wrapped", "latex"]
REAL_IDS = ["id_R_10", "id_R_12", "id_R_13", "id_R_15", "id_R_16", "id_R_17"]


def load_head(ckpt_path, device="cuda:0"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    h = build_head(ck["cfg"]).to(device)
    h.load_state_dict(ck["state_dict"])
    h.eval()
    return h, ck


def subgroup(paths, labels, fake_score, tau, tag_list, cls, name):
    """Per-subgroup recall at a FIXED tau. `cls` is the 3-class label the subgroup belongs to."""
    rows = []
    for t in tag_list:
        m = np.array([t in p for p in paths]) & (labels == cls)
        if not m.any():
            continue
        if cls == M.REAL:
            rec = float((fake_score[m] < tau).mean())          # real recall = correctly kept
        else:
            rec = float((fake_score[m] >= tau).mean())          # fake recall = correctly caught
        rows.append((t, int(m.sum()), rec, float(fake_score[m].mean())))
    if rows:
        print(f"  -- {name} recall @ tau={tau:.6f} --")
        for t, n, r, s in sorted(rows, key=lambda x: x[2]):
            print(f"     {t:<18} n={n:>6,} recall={r:.6f}  mean_score={s:.4f}")
    return rows


def report(head, feats, labels, paths, tau_map, title, device="cuda:0", full=True):
    fs, pr = T.score(head, feats)
    m = M.block(fs, labels, pr)
    print(f"\n=== {title} ===")
    print(f"  n={m['n']:,} (real {m['n_real']:,} / pad {m['n_pad']:,} / deepfake {m['n_deepfake']:,})")
    print(f"  bin_auc={m['bin_auc']:.6f}  ap={m['ap']:.6f}  eer={m['eer']:.6f}")
    print(f"  bal_acc3={m['bal_acc3']:.6f}  acc3={m['acc3_NOT-A-DECISION-METRIC']:.6f} <- NOT-A-DECISION-METRIC")
    print(f"  rec3: real={m['rec3_real']:.6f} pad={m['rec3_pad']:.6f} deepfake={m['rec3_deepfake']:.6f}")
    print(f"  confusion (rows=true real/pad/df, cols=pred): {m['confusion']}")
    # the DEPLOYABLE 3-class decision (threshold, then which-kind-of-fake), reported next to argmax
    # because argmax alone understates an SPC head by construction
    # explicit None checks: `or` would silently fall through on a legitimate tau of exactly 0.0
    tau98 = tau_map.get(98, tau_map.get("98"))
    # A threshold fitted on the very set being reported is NOT a deployable threshold, and labelling it
    # as one is worse than crashing because the number looks right. If no externally-fitted tau was
    # supplied, say so in the label instead of quietly self-fitting under the same name.
    self_fitted = tau98 is None
    if self_fitted:
        tau98 = m["tau@real98"]
    tag = ("SELF-FITTED on THIS set -- NOT DEPLOYABLE, for reference only"
           if self_fitted else "DEPLOYABLE (tau fitted on DEV-A, applied unchanged)")
    at = M.block_at_tau(fs, labels, pr, tau98)
    m.update(at)
    m["tau_self_fitted"] = bool(self_fitted)
    if self_fitted:
        print(f"  WARNING no external threshold was supplied (--fit-dev / --taus), so the 3-class "
              f"decision below is fitted on the set it is reported on.")
    print(f"  3-class @ tau={tau98:.6f} [{tag}]: acc3={at['acc3@tau']:.6f} "
          f"bal_acc3={at['bal_acc3@tau']:.6f} | real={at.get('rec3@tau_real',float('nan')):.6f} "
          f"pad={at.get('rec3@tau_pad',float('nan')):.6f} "
          f"deepfake={at.get('rec3@tau_deepfake',float('nan')):.6f}")
    print(f"  confusion@tau (rows=true, cols=pred): {at['confusion@tau']}")
    # Thresholds FITTED ELSEWHERE (DEV-A), applied here unchanged. These were previously computed as
    # locals and only PRINTED, so the saved json carried nothing but the SELF-FITTED numbers from
    # M.block() -- and a reader (the PDF builder did exactly this) would pair a DEV-A tau with
    # self-fitted recalls and label the result "applied". Persist them under `applied`, keyed by the
    # target, so the two families of number can never be mixed downstream.
    m["applied"] = {}
    for tgt, tau in sorted(tau_map.items()):
        real = fs[labels == M.REAL]; pos = fs[labels != M.REAL]
        ach = float((real < tau).mean())
        fr = float((pos >= tau).mean())
        pd_ = float((fs[labels == M.PAD] >= tau).mean()) if (labels == M.PAD).any() else float("nan")
        df_ = float((fs[labels == M.DEEPFAKE] >= tau).mean()) if (labels == M.DEEPFAKE).any() else float("nan")
        at_t = M.block_at_tau(fs, labels, pr, tau)
        m["applied"][str(tgt)] = {
            "tau": float(tau), "tau_source": "DEV-A (group-disjoint split of the trainset)",
            "real_rec_achieved": ach, "fake_rec": fr, "pad_rec": pd_, "deepfake_rec": df_,
            "acc3@tau": at_t["acc3@tau"], "bal_acc3@tau": at_t["bal_acc3@tau"],
            "rec3@tau_real": at_t.get("rec3@tau_real"), "rec3@tau_pad": at_t.get("rec3@tau_pad"),
            "rec3@tau_deepfake": at_t.get("rec3@tau_deepfake"),
            "confusion@tau": at_t["confusion@tau"]}
        print(f"  tau@real{tgt} (fitted on DEV-A) = {tau:.6f} -> ACHIEVED real_rec={ach:.6f} "
              f"| fake_rec={fr:.6f} (pad {pd_:.6f} / deepfake {df_:.6f}) "
              f"| deployable 3-class acc={at_t['acc3@tau']:.6f} bal={at_t['bal_acc3@tau']:.6f}")
    if not tau_map:
        print("  (no externally-fitted thresholds supplied -> `applied` is empty; every number below "
              "is self-fitted on THIS set and is not a deployable claim)")
    # self-fitted thresholds too, so the transfer gap is visible rather than implied
    print("  [self-fitted on THIS set, for reference only -- not a deployable threshold]")
    for t in (95, 98, 99):
        print(f"     tau@real{t}={m[f'tau@real{t}']:.6f} real_rec={m[f'real_rec@real{t}']:.6f} "
              f"fake_rec={m[f'fake_rec@real{t}']:.6f} (pad {m[f'pad_rec@real{t}']:.6f} / "
              f"deepfake {m[f'deepfake_rec@real{t}']:.6f})")
    if full and paths is not None:
        subgroup(paths, labels, fs, tau98, REAL_IDS, M.REAL, "per-identity REAL")
        rp = np.array(["real_photo__" in p for p in paths]) & (labels == M.REAL)
        if rp.any():
            print(f"     {'real_photo__*':<18} n={int(rp.sum()):>6,} recall={float((fs[rp] < tau98).mean()):.6f}")
        subgroup(paths, labels, fs, tau98, ATTACKS, M.PAD, "per-attack-type PAD")
        for c, nm in ((M.REAL, "real"), (M.PAD, "pad"), (M.DEEPFAKE, "deepfake")):
            s = fs[labels == c]
            if len(s):
                q = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
                print(f"  hist {nm:<9} p1={q[0]:.3f} p5={q[1]:.3f} p25={q[2]:.3f} p50={q[3]:.3f} "
                      f"p75={q[4]:.3f} p95={q[5]:.3f} p99={q[6]:.3f}")
    return m, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache/eval")
    ap.add_argument("--expect", default="eval", choices=["eval", "train", "none"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--title", default=None)
    ap.add_argument("--fit-dev", action="store_true", help="fit tau on DEV-A of the train cache")
    ap.add_argument("--taus", default=None, help="json {\"95\":t,\"98\":t,\"99\":t} to apply as-is")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    head, ck = load_head(a.ckpt, a.device)
    print(f"[eval] ckpt={a.ckpt} cfg={ck['cfg']} params={head.n_trainable():,}")

    tau_map = {}
    if a.taus:
        tau_map = {int(k): float(v) for k, v in json.loads(a.taus).items()}
    elif a.fit_dev:
        b = T.load_bundle(device=a.device)
        md = T.evaluate_split(head, b, 1)
        tau_map = {t: md[f"tau@real{t}"] for t in (95, 98, 99)}
        print(f"[eval] DEV-A fit: " + " ".join(
            f"tau@real{t}={tau_map[t]:.6f}(achieved {md[f'real_rec@real{t}']:.6f})" for t in (95, 98, 99)))
        print(f"[eval] DEV-A block: {M.fmt(md)}")
        del b
        torch.cuda.empty_cache()

    exp = {"eval": cache_io.EXPECT_EVAL, "train": cache_io.EXPECT_TRAIN, "none": None}[a.expect]
    c = cache_io.load_shards(os.path.join(ROOT, a.cache))
    cache_io.verify(c, exp, tag=os.path.basename(a.cache))
    # the head, the prototypes it was initialised from, and these features must share one encoder
    _pf = None
    _pp = os.path.join(ROOT, "cache/prototypes.pt")
    if os.path.exists(_pp):
        try:
            _pf = torch.load(_pp, map_location="cpu", weights_only=False).get("fingerprint")
        except Exception:
            _pf = None
    cache_io.assert_same_encoder(c.get("prov"), ck.get("fingerprint"), _pf, tag="eval")
    feats = torch.from_numpy(c["feats"]).to(a.device)
    m, fs = report(head, feats, c["labels"], c["paths"].tolist(), tau_map,
                   a.title or os.path.basename(a.cache), a.device)
    if a.save:
        json.dump({"ckpt": a.ckpt, "cache": a.cache, "tau_map": tau_map,
                   "key_semantics": {
                       "top-level tau@realN / real_rec@realN / fake_rec@realN / pad_rec@realN / "
                       "deepfake_rec@realN": "SELF-FITTED on this cache -- reference only",
                       "top-level bal_acc3 / acc3_NOT-A-DECISION-METRIC / rec3_*": "RAW ARGMAX, which "
                       "structurally understates an SPC head (max softmax ~0.44) -- not a decision",
                       "applied.<target>.*": "threshold fitted on DEV-A and applied here unchanged -- "
                       "this is the deployable family, including bal_acc3@tau and rec3@tau_*"},
                   "metrics": {k: v for k, v in m.items()}}, open(a.save, "w"), indent=1)
        np.save(a.save.replace(".json", "_scores.npy"), fs)
        print(f"[eval] -> {a.save}")


if __name__ == "__main__":
    main()

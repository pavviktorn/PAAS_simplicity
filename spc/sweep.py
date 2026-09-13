"""Selection sweep, all on DEV-A, in the order fixed before any numbers were seen:
   stage 1 prompts (+ C1 class-mean and C2 random controls)  ->  stage 2 hyperparameters
   ->  stage 3 head topology  ->  stage 4 class imbalance.

C1/C2 are not decoration. With 1.25 M training images the initialisation may wash out entirely; if
every text triplet lands within noise of C2 (random), the honest finding is "the prompt does not
matter at this data scale" and this script has to be able to say that instead of manufacturing a
prompt story. C1 (class means, no text at all) is the baseline the text tower has to beat to justify
itself. H4 (MLP) is reported as the CEILING of the frozen features -- if it dominates, the head is
the bottleneck, not PE, and that is worth more than a tuned number.

Primary metric: DEV-A fake_recall@real98. Tie-break within 0.002: bin_auc, then shorter prompts.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M                                   # noqa: E402
import train_spc as T                                 # noqa: E402
from text_prototypes import TRIPLETS                  # noqa: E402

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
KEY = "fake_rec@real98"
TIE = 0.002
# DEV is contaminated by the trainset's internal redundancy (measured: 3.84% of path-distinct DEV-A
# images are bit-identical to a TRAIN image, 11.35% within cosine 0.99, median 0.9565). c99 is the
# primary level; the other two exist to test whether the CHOICE depends on the contamination.
# ORDER MATTERS: run_one passes these to train(), which uses levels[0] as the level it tracks its
# best checkpoint on and returns as the row's top-level metric. With "raw" first, every selection would
# have been made on the CONTAMINATED level while the report claimed c99. PRIMARY must be first.
PRIMARY = "c99"
LEVELS = (PRIMARY, "raw", "c95")
DISPLAY = ("raw", "c99", "c95")


def _jsonable(o):
    """`default=str` would silently turn a numpy float into the STRING "0.98..." -- and the report's
    f6() swallows unparseable values as "-", so a real measurement could disappear from the PDF with
    no error anywhere. Convert what is convertible; raise on anything else."""
    import numpy as _np
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(f"[sweep] refusing to serialise {type(o).__name__} into sweep.json: {o!r}")


def prompt_len(name):
    return sum(len(s) for s in TRIPLETS[name]) if name in TRIPLETS else 999


def must_pick(rows, stage):
    """pick() but never returns None: if every row is leak-suspect or NaN there is nothing to select
    and the sweep must say so loudly instead of dying later on `None[...]`."""
    w = pick(rows)
    if w is not None:
        return w
    # Every row is saturated and/or non-finite. Falling back is correct here rather than aborting:
    # bin_auc saturating for ALL configs is exactly the case fake_rec@real98 was chosen as the primary
    # metric to handle. Say so explicitly so the report cannot present this as a clean selection.
    fin = [r for r in rows if np.isfinite(r.get(KEY, float("nan")))]
    if not fin:
        raise SystemExit(
            f"[sweep] stage '{stage}': all {len(rows)} rows have a non-finite {KEY}. There is nothing "
            f"to select and picking anyway would be inventing a result.")
    fin.sort(key=lambda r: (-r[KEY], -r.get("bin_auc", 0), prompt_len(r.get("prompts", ""))))
    print(f"[sweep] stage '{stage}': ALL {len(rows)} configs saturate DEV bin_auc >= {0.9999}. "
          f"Selecting on {KEY} alone ({fin[0].get('name')} = {fin[0][KEY]:.6f}); the DEV split cannot "
          f"discriminate on AUC, so this choice is weakly determined and is reported as such.",
          flush=True)
    return fin[0]


def pick(rows):
    """The fixed selection rule. Leak-suspect rows can never win."""
    ok = [r for r in rows if not r.get("leak_suspect") and np.isfinite(r.get(KEY, float("nan")))]
    if not ok:
        return None
    top = max(r[KEY] for r in ok)
    cand = [r for r in ok if r[KEY] >= top - TIE]
    cand.sort(key=lambda r: (-r["bin_auc"], prompt_len(r.get("prompts", ""))))
    return cand[0]


def spearman(a, b):
    """Rank correlation without scipy. a,b are score lists over the SAME candidates."""
    def rk(x):
        o = np.argsort(np.argsort(-np.asarray(x, dtype=float)))
        return o.astype(float)
    ra, rb = rk(a), rk(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def pick_at(rows, level):
    """Apply the REAL selection rule (pick: 0.002 tie band, then bin_auc, then prompt length, with
    saturated rows deprioritised) to a given contamination level, by projecting each row's metrics for
    that level into the fields pick() reads. The previous stability check used a plain nanargmax, so it
    could report "stable" for a level whose actual winner under the tie-break differs -- i.e. it tested
    something other than the decision being made."""
    proj = []
    for r in rows:
        lv = (r.get("_levels") or {}).get(level)
        if not lv:
            continue
        rr = dict(r)
        rr[KEY] = lv.get(KEY, float("nan"))
        rr["bin_auc"] = lv.get("bin_auc", float("nan"))
        rr["leak_suspect"] = bool(np.isfinite(rr["bin_auc"]) and rr["bin_auc"] >= T.LEAK_AUC)
        proj.append(rr)
    return pick(proj)


def stability(rows, label):
    """Does the DECISION survive de-contamination? Compares the winner that pick() would choose at
    each level -- not an argmax -- because pick() is what actually selects."""
    named = [r for r in rows if r.get("_levels")]
    if len(named) < 2:
        return {}
    out = {"label": label, "n_candidates": len(named), "top1": {}, "spearman": {}}
    per = {lv: [r["_levels"].get(lv, {}).get(KEY, float("nan")) for r in named] for lv in DISPLAY}
    for lv in DISPLAY:
        w = pick_at(named, lv)
        out["top1"][lv] = w.get("name") if w else None
    for lv in ("c99", "c95"):
        out["spearman"][f"raw~{lv}"] = spearman(per["raw"], per[lv])
    out["spearman"]["c99~c95"] = spearman(per["c99"], per["c95"])
    agree = len({x for x in out["top1"].values() if x}) == 1
    out["top1_agree"] = agree
    print(f"\n[stability] {label}: winner-under-the-selection-rule per level {out['top1']} | "
          f"spearman {({k: round(v,4) for k,v in out['spearman'].items()})} | "
          f"{'STABLE (de-contamination does not change the choice)' if agree else 'UNSTABLE -- the choice depends on the contamination level, so it is NOT trusted and the paper default is used instead'}",
          flush=True)
    return out


def run_one(cfg, b, protos, verbose=False, levels=(PRIMARY,)):
    t0 = time.time()
    row = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if cfg.get("_zeroshot"):
        _, m = T.zeroshot(cfg, b, protos, levels=levels)
    else:
        _, m = T.train(cfg, b, protos, verbose=verbose, levels=levels)
    # keep the NUMBERS even when saturated: an excluded row with no metrics cannot be reported, and
    # if every row saturates the sweep still has to select something and say why.
    row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
    row["_levels"] = m.get("_levels", {})
    if m.get("_leak_suspect"):
        row["leak_suspect"] = True
        print(f"  !! saturated DEV bin_auc for {cfg.get('name')} "
              f"(deprioritised in selection, numbers kept)", flush=True)
    row["secs"] = round(time.time() - t0, 1)
    return row


def show(rows, title, keys=(KEY, "bin_auc", "ap", "eer", "fake_rec@real99", "rec3_deepfake")):
    print(f"\n########## {title} ##########", flush=True)
    hdr = f"{'name':<26}" + "".join(f"{k:>18}" for k in keys) + f"{'params':>9}{'s':>7}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -(r.get(KEY) if np.isfinite(r.get(KEY, float('nan'))) else -9)):
        flag = " LEAK?" if r.get("leak_suspect") else ""
        print(f"{r.get('name','?'):<26}" + "".join(
            f"{r.get(k, float('nan')):>18.6f}" if isinstance(r.get(k), float) else f"{'-':>18}"
            for k in keys) + f"{r.get('_params',0):>9,}{r.get('secs',0):>7.1f}{flag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "prompts", "hparams", "heads", "imbalance", "trainset"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/spc/sweep.json"))
    ap.add_argument("--default-out", default=os.path.join(ROOT, "configs/default.json"),
                    help="where to write the selected config. The rehearsal MUST point this somewhere "
                         "else: the sweep previously always wrote configs/default.json, so a 2%-data "
                         "1-epoch rehearsal silently overwrote the production default.")
    a = ap.parse_args()

    b = T.load_bundle(device=a.device)
    P = torch.load(os.path.join(ROOT, "cache/prototypes.pt"), weights_only=False)["protos"]
    protos = {k: v.to(a.device) for k, v in P.items()}
    cm = T.class_means(b, a.device)
    base = {"lr": 2e-5, "batch_size": 128, "epochs": a.epochs, "weight_decay": 0.0,
            "schedule": "constant", "class_weight": "none", "train_frac": a.train_frac, "seed": 0}
    results = {}

    # ---- stage 1: prompts, zero-shot AND post-calibration -------------------------------------
    names = list(TRIPLETS) + ["T10", "C1", "C2"]
    if a.stage in ("all", "prompts"):
        zs = []
        for n in names:
            cfg = {**base, "name": f"zs_{n}", "head": "H0", "prompts": n, "_zeroshot": True,
                   "_class_means": cm}
            zs.append(run_one(cfg, b, protos, levels=LEVELS))
        show(zs, "STAGE 1a  prompts, ZERO-SHOT (prototypes as initialised, no training)")
        results["prompts_zeroshot"] = zs

        cal = []
        for n in names:
            cfg = {**base, "name": f"cal_{n}", "head": "H0", "prompts": n, "_class_means": cm}
            cal.append(run_one(cfg, b, protos, levels=LEVELS))
        show(cal, "STAGE 1b  prompts, POST-CALIBRATION (paper recipe: H0 lr2e-5 bs128 2ep)")
        results["prompts_calibrated"] = cal
        st = stability(cal, "prompt selection")
        results["stability_prompts"] = st
        _w = must_pick(cal, "1b prompts")
        best_prompt = _w["prompts"]
        fell_back = bool(st) and not st.get("top1_agree")
        if fell_back:
            # This is the documented rule, now actually executed: a selection that flips with the
            # contamination level is noise-dominated, so the paper's own wording is used instead.
            print(f"[sweep] prompt selection is UNSTABLE -> falling back to T0 (paper wording) "
                  f"instead of the c99 winner {best_prompt!r}", flush=True)
            best_prompt = "T0"
        results["winners"] = {"prompts": {"name": _w.get("name"), "prompts": best_prompt,
                                          "c99_winner": _w.get("prompts"),
                                          "fell_back_to_paper": fell_back,
                                          KEY: _w.get(KEY), "bin_auc": _w.get("bin_auc")}}
        print(f"\n>>> stage 1 winner: prompts={best_prompt}", flush=True)
    else:
        best_prompt = "T0"

    # ---- stage 2: hyperparameters --------------------------------------------------------------
    if a.stage in ("all", "hparams"):
        hp = []
        for lr in (2e-5, 1e-4, 3e-4, 1e-3):
            for ls in (False, True):
                cfg = {**base, "name": f"lr{lr:g}_{'scale' if ls else 'noscale'}",
                       "head": "H1" if ls else "H0", "prompts": best_prompt,
                       "learn_scale": ls, "lr": lr, "_class_means": cm}
                hp.append(run_one(cfg, b, protos, levels=LEVELS))
        show(hp, "STAGE 2a  lr x learnable-scale")
        w = must_pick(hp, "2a lr x scale"); best_lr, best_ls = w["lr"], w["learn_scale"]
        hp2 = []
        for ep in (1, 2, 5):
            for wd in (0.0, 1e-4, 1e-2):
                for sch in ("constant", "cosine"):
                    cfg = {**base, "name": f"ep{ep}_wd{wd:g}_{sch}", "head": "H1" if best_ls else "H0",
                           "prompts": best_prompt, "learn_scale": best_ls, "lr": best_lr,
                           "epochs": ep, "weight_decay": wd, "schedule": sch, "_class_means": cm}
                    hp2.append(run_one(cfg, b, protos, levels=LEVELS))
        show(hp2, "STAGE 2b  epochs x weight-decay x schedule")
        results["hparams"] = hp + hp2
        # the learning rate is a selection decision like any other, so it gets the same test
        results["stability_hparams"] = stability(hp + hp2, "hyperparameters")
        w2 = must_pick(hp + hp2, "2b epochs x wd x sched")
        best = {k: w2.get(k, base.get(k)) for k in
                ("lr", "learn_scale", "epochs", "weight_decay", "schedule")}
        hp_fell_back = bool(results.get("stability_hparams")) and \
            not results["stability_hparams"].get("top1_agree")
        if hp_fell_back:
            paper_hp = {"lr": 2e-5, "learn_scale": False, "epochs": 2, "weight_decay": 0.0,
                        "schedule": "constant"}
            print(f"[sweep] hyperparameter selection is UNSTABLE -> falling back to the paper recipe "
                  f"{paper_hp} instead of {best}", flush=True)
            best = paper_hp
        results.setdefault("winners", {})["hparams"] = {
            "name": w2.get("name"), **best, "fell_back_to_paper": hp_fell_back,
            KEY: w2.get(KEY), "bin_auc": w2.get("bin_auc")}
        print(f"\n>>> stage 2 winner: {best}", flush=True)
    else:
        best = {"lr": 2e-5, "learn_scale": False, "epochs": a.epochs, "weight_decay": 0.0,
                "schedule": "constant"}

    # ---- stage 3: head topology ---------------------------------------------------------------
    if a.stage in ("all", "heads"):
        hd = []
        for spec in ({"head": "H0", "name": "H0_linear", "learn_scale": False},
                     {"head": "H1", "name": "H1_scale", "learn_scale": True},
                     {"head": "H2", "name": "H2_cosine", "cosine": True, "learn_scale": True},
                     {"head": "H3", "name": "H3_k4", "k": 4, "prompts": "K4", "learn_scale": True},
                     # H4 gets its OWN lr: it has 657k params, not 3.8k, and inheriting the
                     # prototype lr would underfit it and UNDERSTATE the ceiling -- which would turn
                     # "the features are the bottleneck" into an artefact of a borrowed lr.
                     {"head": "H4", "name": "H4_mlp_DIAGNOSTIC", "lr": 1e-3, "epochs": 2}):
            cfg = {**base, **best, "prompts": best_prompt, "_class_means": cm, **spec}
            hd.append(run_one(cfg, b, protos, levels=LEVELS))
        show(hd, "STAGE 3  head topology (H4 = frozen-feature CEILING, not deployable)")
        results["heads"] = hd
        st_h = stability([r for r in hd if r.get("head") != "H4"], "head topology")
        results["stability_heads"] = st_h
        dep = [r for r in hd if r["head"] != "H4"]
        wh = must_pick(dep, "3 head topology")
        head_fell_back = bool(st_h) and not st_h.get("top1_agree")
        if head_fell_back:
            h0 = next((r for r in dep if r.get("head") == "H0"), None)
            print(f"[sweep] head selection is UNSTABLE -> falling back to H0 (the paper's linear head) "
                  f"instead of {wh.get('name')!r}", flush=True)
            if h0 is not None:
                wh = h0
        results.setdefault("winners", {})["head"] = {
            "name": wh.get("name"), "head": wh.get("head"), KEY: wh.get(KEY),
            "fell_back_to_paper": head_fell_back,
            "bin_auc": wh.get("bin_auc"), "params": wh.get("_params")}
        print(f"\n>>> stage 3 winner (deployable): {wh['name']}", flush=True)
        h4 = [r for r in hd if r["head"] == "H4"]
        if h4 and np.isfinite(h4[0].get(KEY, float("nan"))):
            results["winners"]["h4_ceiling"] = {
                KEY: h4[0][KEY], "gap_vs_best_deployable": h4[0][KEY] - wh[KEY],
                # DEV-ONLY, and NOT a conclusion. Measured afterwards on the clean testset: H4's
                # DEV-A advantage of +0.047801 became +0.000050 on fr@real98 and -0.003824 on bin_auc,
                # with deepfake recall 5 points WORSE. A 657k-param MLP exploits the near-duplicates
                # DEV still contains (58% within cosine 0.95 of a TRAIN image) in a way a 3.8k linear
                # head cannot, so a high-capacity probe is an unreliable ceiling estimator here. The
                # verdict must come from the testset.
                "verdict_DEV_ONLY_NOT_A_CONCLUSION": (
                    "H4 leads on DEV by >0.01; this must be confirmed on the clean testset before it "
                    "means anything, because DEV contamination favours high-capacity heads"),
                "confirm_on": "runs/spc/testset_h4.json vs runs/spc/testset.json"}
            print(f">>> H4 ceiling {KEY}={h4[0][KEY]:.6f} vs best deployable {wh[KEY]:.6f} "
                  f"(gap {h4[0][KEY]-wh[KEY]:+.6f}) -> "
                  f"{'H4 leads on DEV -- CONFIRM ON THE TESTSET before concluding anything (DEV favours capacity)' if h4[0][KEY]-wh[KEY] > 0.01 else 'no DEV advantage for extra capacity'}",
                  flush=True)
    else:
        wh = {"head": "H1", "learn_scale": True, "name": "H1_scale"}

    # ---- stage 4: class imbalance -------------------------------------------------------------
    if a.stage in ("all", "imbalance"):
        im = []
        for cw in ("none", "inv", "sqrtinv", [1.5, 1, 1], [2, 1, 1], [3, 1, 1]):
            cfg = {**base, **best, "prompts": best_prompt, "_class_means": cm,
                   **{k: v for k, v in wh.items() if k in ("head", "k", "cosine", "learn_scale")},
                   "class_weight": cw, "name": f"cw_{cw if isinstance(cw,str) else 'r%g'%cw[0]}"}
            if cfg.get("head") == "H3":
                cfg["prompts"] = "K4"
            im.append(run_one(cfg, b, protos, levels=LEVELS))
        show(im, "STAGE 4  class imbalance")
        results["imbalance"] = im
        results["stability_imbalance"] = stability(im, "class imbalance")
        wi = must_pick(im, "4 imbalance")
        base_row = next((r for r in im if r.get("name") == "cw_none"), None)
        if base_row is None or not np.isfinite(base_row.get(KEY, float("nan"))):
            raise SystemExit("[sweep] stage 4 has no usable 'cw_none' baseline row -- the >0.005 "
                             "adoption bar is defined RELATIVE to it, so adopting anything here would "
                             "be comparing against nothing.")
        im_fell_back = bool(results.get("stability_imbalance")) and \
            not results["stability_imbalance"].get("top1_agree")
        take = (wi[KEY] - base_row[KEY] > 0.005) and not im_fell_back
        if im_fell_back:
            print("[sweep] imbalance selection is UNSTABLE -> keeping 'none' (the paper baseline)",
                  flush=True)
        print(f"\n>>> stage 4: best={wi['name']} ({wi[KEY]:.6f}) vs none ({base_row[KEY]:.6f}) "
              f"delta={wi[KEY]-base_row[KEY]:+.6f} -> "
              f"{'ADOPT' if take else 'KEEP none (delta <= 0.005, a mean-1 reweight is ~a boundary shift the threshold fit undoes)'}",
              flush=True)
        results.setdefault("winners", {})["imbalance"] = {
            "name": wi.get("name"), "class_weight": wi.get("class_weight"), KEY: wi.get(KEY),
            "none_baseline": base_row.get(KEY), "delta": wi.get(KEY) - base_row.get(KEY),
            "adopted": bool(take), "bar": 0.005, "fell_back_to_paper": im_fell_back}
        final_cw = wi["class_weight"] if take else "none"
    else:
        final_cw = "none"

    # ---- stage 5: trainset composition -------------------------------------------------------
    final_arm = "as_given"
    if a.stage in ("all", "trainset"):
        ta = []
        for armname in ("as_given", "drop_conflicts", "dedup_content"):
            cfg = {**base, **best, "prompts": best_prompt, "_class_means": cm,
                   **{k: v for k, v in wh.items() if k in ("head", "k", "cosine", "learn_scale")},
                   "class_weight": final_cw, "train_data": armname, "name": f"data_{armname}"}
            if cfg.get("head") == "H3":
                cfg["prompts"] = "K4"
            ta.append(run_one(cfg, b, protos, levels=LEVELS))
        show(ta, "STAGE 5  trainset composition (duplicates / contradictory labels)")
        results["trainset"] = ta
        results["stability_trainset"] = stability(ta, "trainset composition")
        wt = must_pick(ta, "5 trainset")
        asg = next((r for r in ta if r.get("name") == "data_as_given"), None)
        ta_fell_back = bool(results["stability_trainset"]) and \
            not results["stability_trainset"].get("top1_agree")
        final_arm = "as_given" if ta_fell_back else wt.get("train_data", "as_given")
        results.setdefault("winners", {})["trainset"] = {
            "name": wt.get("name"), "train_data": final_arm,
            KEY: wt.get(KEY), "as_given_baseline": (asg or {}).get(KEY),
            "delta_vs_as_given": (wt.get(KEY) - asg.get(KEY)) if asg else None,
            "fell_back_to_paper": ta_fell_back}
        print(f"\n>>> stage 5 winner: train_data={final_arm} "
              f"(as_given {(asg or {}).get(KEY)})", flush=True)

    default = {"name": "default", **base, **best, "prompts": best_prompt, "train_data": final_arm,
               **{k: v for k, v in wh.items() if k in ("head", "k", "cosine", "learn_scale")},
               "class_weight": final_cw, "train_frac": 1.0}
    if default.get("head") == "H3":
        default["prompts"] = "K4"
    default.pop("_class_means", None)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1, default=_jsonable)
    dcfg = a.default_out
    json.dump(default, open(dcfg, "w"), indent=1)
    print(f"\n>>> DEFAULT CONFIG (DEV-A selected) -> {dcfg}\n{json.dumps(default, indent=1)}", flush=True)
    print(f">>> sweep rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

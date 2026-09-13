"""Train the PE-SPC head on cached frozen features.

The encoder never runs here. One config = a few seconds of matmuls on a 3.3 GB resident cache, which
is what makes the prompt/hyperparameter search in sweep.py honest instead of a story.

Selection discipline (fixed before any numbers were seen):
  * everything is selected on DEV-A, the winner is confirmed ONCE on DEV-B, the testset selects
    nothing -- not the prompt, not the lr, not the threshold;
  * the headline metric is fake_recall@real98, not AUC (AUC saturates >0.99 for every reasonable
    config here and cannot rank candidates) and never acc@0.5 (structurally meaningless for this
    head: max softmax ~= 0.44);
  * any DEV bin_auc >= 0.9999 aborts as a suspected leak instead of being saved as `best`.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache_io, metrics as M, split_dev            # noqa: E402
from head import build_head, BIAS_INIT              # noqa: E402

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
LEAK_AUC = 0.9999


class LeakSuspect(RuntimeError):
    """Raised when a DEV metric is too saturated to rank anything. Catchable ON PURPOSE: one
    suspicious config must not kill a 30-config sweep, but it must never be silently recorded as a
    win either -- sweep.py flags it and refuses to select it."""


def load_bundle(prefix="cache/train", device="cuda:0", verify=True, expect=cache_io.EXPECT_TRAIN):
    c = cache_io.load_shards(os.path.join(ROOT, prefix))
    if verify:
        cache_io.verify(c, expect, tag=os.path.basename(prefix))
    sp = json.load(open(os.path.join(ROOT, "manifests/split.json")))
    big = set(sp["big_groups"])
    t0 = time.time()
    paths_l = c["paths"].tolist()
    split = split_dev.assign(paths_l, big, sp["seed"])
    print(f"[data] split assigned in {time.time()-t0:.1f}s "
          f"train={int((split==0).sum()):,} devA={int((split==1).sum()):,} devB={int((split==2).sum()):,}",
          flush=True)
    # The manifest contains DUPLICATE RECORDS: 41,557 paths appear exactly 10x and 16 appear 16x, so
    # 1,360,956 records cover only 986,703 distinct images (27.5% redundant, almost all from
    # fmt_error11). All copies of a path land in the same split (_u01 is a pure function of the group
    # key), so this is not a leak -- but a duplicated DEV image would otherwise count TEN TIMES in the
    # selection metric, letting 4k images carry the weight of 40k. DEV metrics therefore run on FIRST
    # OCCURRENCES only. Training keeps the duplicates on purpose: GSD/SeLop/mids9c were trained on this
    # same manifest, so removing them here would change the trainset relative to the models we compare
    # against.
    seen = set()
    uniq = np.zeros(len(paths_l), dtype=bool)
    for i, p in enumerate(paths_l):
        if p not in seen:
            seen.add(p); uniq[i] = True

    # Contamination artifacts (spc/precompute_contam.py). Path-dedup alone is NOT enough: the same
    # IMAGE appears under different paths, so only 940,299 of 1,360,956 rows are distinct content and
    # 3.84% of path-distinct DEV-A images are BIT-IDENTICAL to a TRAIN image (11.35% within cos 0.99,
    # median 0.9565). 285 content clusters also carry contradictory labels and are unscoreable.
    # DEV is therefore evaluated at THREE levels and selection is only trusted if the ranking is
    # stable across them -- see sweep.py. The testset needs none of this: it is clean against the same
    # TRAIN set (0 bit-identical, 0 within cos 0.99, median 0.8330).
    cpath = os.path.join(ROOT, "cache/dev_contam.npz")
    maxcos = exact_dup = conflict = None
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=False)
        want = {"paths_sha256": hashlib.sha256("\n".join(paths_l).encode()).hexdigest(),
                "split_json_sha256": cache_io.sha256(os.path.join(ROOT, "manifests/split.json")),
                "split_seed": int(sp["seed"]),
                "groupkey_version": int(getattr(split_dev, "GROUPKEY_VERSION", 0))}
        got = json.loads(str(z["fingerprint"])) if "fingerprint" in z.files else None
        if got is None:
            raise SystemExit(
                f"[data] {cpath} predates the fingerprint (and therefore also predates the fix to the "
                f"exact-duplicate statistic, which used a raw fp16 dot and under-counted duplicates by "
                f"~40%). Delete it and re-run spc/precompute_contam.py.")
        bad = [k for k, v in want.items() if got.get(k) != v]
        if bad:
            det = "; ".join(f"{k}: artifact={str(got.get(k))[:16]!r} live={str(want[k])[:16]!r}"
                            for k in bad)
            raise SystemExit(
                f"[data] {cpath} was built under a different {bad} ({det}). Its masks are POSITIONAL, "
                f"so applying them now would score the wrong rows -- and a changed split.json, seed or "
                f"grouping version changes which rows are DEV at all. Re-run "
                f"spc/precompute_contam.py.")
        if len(z["max_cos_to_train"]) != len(paths_l):
            raise SystemExit(f"[data] {cpath} has {len(z['max_cos_to_train']):,} rows, cache has "
                             f"{len(paths_l):,}")
        maxcos = z["max_cos_to_train"]; conflict = z["conflict"]
        exact_dup = z["exact_dup_in_train"]
        content_id = z["content_id"] if "content_id" in z.files else None
    elif os.environ.get("ALLOW_CONTAMINATED_DEV") == "1":
        # Explicit, loud, opt-in. The previous behaviour filled max_cos with -1 and carried on, which
        # made every level pass the filter -- so a run WITHOUT the artifact silently reported
        # contaminated DEV numbers under the label "c99". A label that can silently mean its opposite
        # is worse than a crash.
        print(f"[data] ALLOW_CONTAMINATED_DEV=1: no {os.path.basename(cpath)}; DEV metrics will be RAW "
              f"and the level names are meaningless. NOT valid for selection or for the report.",
              flush=True)
        maxcos = np.full(len(paths_l), -1.0, np.float32)
        exact_dup = np.zeros(len(paths_l), bool)
        conflict = np.zeros(len(paths_l), bool)
        content_id = None
    else:
        raise SystemExit(
            f"[data] missing {cpath}. Without it the de-contaminated DEV levels cannot be computed, "
            f"and the previous behaviour was to report RAW numbers under the name 'c99'. Run\n"
            f"    {sys.executable} spc/precompute_contam.py\n"
            f"or set ALLOW_CONTAMINATED_DEV=1 to accept raw DEV explicitly (not valid for selection).")
    for nm, k in (("devA", 1), ("devB", 2)):
        m = split == k
        base = m & uniq & ~conflict
        print(f"[data] {nm}: {int(m.sum()):,} rows -> {int((m & uniq).sum()):,} distinct paths -> "
              f"{int(base.sum()):,} scoreable (minus {int((m & uniq & conflict).sum()):,} "
              f"contradictory-label) | exact TRAIN dups {int((base & exact_dup).sum()):,} "
              f"({(base & exact_dup).sum()/max(base.sum(),1)*100:.2f}%) | "
              f"cos>=0.99 {int((base & (maxcos >= 0.99)).sum()):,} "
              f"cos>=0.95 {int((base & (maxcos >= 0.95)).sum()):,}", flush=True)
    b = {"feats": torch.from_numpy(c["feats"]).to(device),
         "labels": torch.from_numpy(c["labels"]).to(device),
         "split": torch.from_numpy(split.astype(np.int64)).to(device),
         "uniq": torch.from_numpy(uniq).to(device),
         "maxcos": torch.from_numpy(np.asarray(maxcos)).to(device),
         "exact_dup": torch.from_numpy(np.asarray(exact_dup)).to(device),
         "content_id": (torch.from_numpy(np.asarray(content_id).astype(np.int64)).to(device)
                        if content_id is not None else None),
         "conflict": torch.from_numpy(np.asarray(conflict)).to(device),
         "paths": c["paths"], "device": device, "split_meta": sp,
         "n_distinct": int(uniq.sum())}
    return b


def _weights(kind, labels, device):
    if kind in (None, "none", False):
        return None
    n = torch.bincount(labels, minlength=3).float()
    if kind == "inv":
        w = n.sum() / (3 * n)
    elif kind == "sqrtinv":
        w = (n.sum() / (3 * n)).sqrt()
    elif isinstance(kind, (list, tuple)):
        w = torch.tensor([float(x) for x in kind], device=n.device)
    else:
        raise SystemExit(f"[train] unknown class_weight {kind!r}")
    w = w / w.mean()
    return w.to(device)


@torch.no_grad()
def score(head, feats, bs=200_000):
    """-> (fake_score, probs) as numpy. fake_score = 1 - softmax[:, REAL], bit-identical to infer."""
    fs, pr = [], []
    for i in range(0, len(feats), bs):
        z = head(feats[i:i + bs].float())
        p = z.softmax(-1)
        fs.append((1.0 - p[:, M.REAL]).float().cpu()); pr.append(p.float().cpu())
    return torch.cat(fs).numpy().astype(np.float64), torch.cat(pr).numpy()


# DEV contamination levels. `raw` keeps every distinct-path image; `c99`/`c95` additionally drop
# images whose nearest TRAIN neighbour is within that cosine. c99 is the PRIMARY selection level:
# above 0.99 an image is almost certainly the same picture re-encoded, whereas the 0.95 band is
# dominated by genuinely distinct frames from the same capture rig -- cutting there removes 58% of
# DEV-A and leaves only its least-typical tail, which is a worse basis for ranking, not a better one.
# DEV contamination levels. `raw` keeps every distinct-path, scoreable image. c99/c95 additionally
# drop images that are EXACT hash-duplicates of a TRAIN image, plus those whose TRUE cosine to the
# nearest TRAIN image reaches the threshold. Exact duplicates are removed by hash identity at every
# de-contaminated level rather than by a cosine cut, because the fp16 cache makes a raw dot of two
# identical rows equal ||v||^2 in [0.9913, 1.0098] -- a >=0.9999 cut on that misses ~40% of them.
# c99 is PRIMARY: above 0.99 an image is almost certainly the same picture re-encoded, whereas the
# 0.95 band is dominated by genuinely distinct frames from the same capture rig -- cutting there
# removes ~58% of DEV-A and leaves only its least-typical tail, a worse basis for ranking.
LEVELS = {"raw": None, "c99": 0.99, "c95": 0.95}


def dev_mask(b, which, level="c99", dedup=True):
    m = b["split"] == which
    if dedup and "uniq" in b:
        m = m & b["uniq"]
    if "conflict" in b:
        m = m & ~b["conflict"]                      # contradictory labels are never scoreable
    thr = LEVELS.get(level)
    if thr is not None:
        if "exact_dup" in b:
            m = m & ~b["exact_dup"]                 # hash-identical, independent of any threshold
        if "maxcos" in b:
            m = m & (b["maxcos"] < thr)
    return m


def evaluate_split(head, b, which, targets=(0.95, 0.98, 0.99), dedup=True, level="c99"):
    """DEV metrics on distinct, scoreable, decontaminated images. See dev_mask/LEVELS."""
    m = dev_mask(b, which, level, dedup)
    fs, pr = score(head, b["feats"][m])
    return M.block(fs, b["labels"][m].cpu().numpy(), pr, targets)


def make_head(cfg, protos, device):
    h = build_head(cfg)
    name = cfg.get("prompts", "T0")
    if cfg.get("head") != "H4":
        if name == "C2":                                    # random control
            g = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
            p = F.normalize(torch.randn(h.proto.shape, generator=g), dim=-1)
        elif name == "C1":
            p = cfg["_class_means"]                          # filled in by the caller
        elif cfg.get("k", 1) > 1:
            p = protos["K4"] if name in ("K4", "T0") else protos[name].repeat_interleave(cfg["k"], 0)
        else:
            p = protos[name]
        h.init_prototypes(p.float())
    return h.to(device)


def train(cfg, b, protos, verbose=True, dev_which=1, levels=("c99",)):
    device = b["device"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    head = make_head(cfg, protos, device)

    # TRAINSET ARM. Keeping the manifest as-given preserves comparability with GSD/SeLop/mids9c, which
    # trained on this same manifest -- but it is NOT the accuracy-maximising choice, and that is a
    # measurable question rather than a matter of taste. Measured on this cache: TRAIN as-given is
    # 1,260,666 rows at real/pad/deepfake 30.9/40.9/28.1, while one row per distinct IMAGE is 874,025
    # rows at 34.2/36.8/29.0 -- so the duplication silently over-weights PAD by ~4 points, and 34.0% of
    # TRAIN rows sit in a repeated-content cluster. Separately, 1,645 rows across 781 paths carry
    # CONTRADICTORY labels: the identical image pushed toward pad by one row and deepfake by another,
    # i.e. directly opposing gradients on the same input.
    arm = cfg.get("train_data", "as_given")
    tr_mask = b["split"] == 0
    if arm in ("drop_conflicts", "dedup_content"):
        if "conflict" in b:
            tr_mask = tr_mask & ~b["conflict"]
    if arm == "dedup_content":
        if b.get("content_id") is None:
            raise SystemExit("[train] train_data='dedup_content' needs content_id in "
                             "cache/dev_contam.npz -- re-run spc/precompute_contam.py.")
        idx = torch.nonzero(tr_mask, as_tuple=True)[0]
        cid = b["content_id"][idx]
        # keep the FIRST row of each distinct content within TRAIN
        _, first = np.unique(cid.cpu().numpy(), return_index=True)
        keep = idx[torch.from_numpy(np.sort(first)).to(idx.device)]
        tr_mask = torch.zeros_like(tr_mask)
        tr_mask[keep] = True
    elif arm not in ("as_given", "drop_conflicts"):
        raise SystemExit(f"[train] unknown train_data arm {arm!r}")
    tr = torch.nonzero(tr_mask, as_tuple=True)[0]
    if cfg.get("train_frac", 1.0) < 1.0:                     # sweep throughput knob, stated in output
        g = torch.Generator(device=device).manual_seed(0)
        tr = tr[torch.randperm(len(tr), generator=g, device=device)[:int(len(tr) * cfg["train_frac"])]]
    y_all = b["labels"]
    w = _weights(cfg.get("class_weight", "none"), y_all[tr], device)

    epochs, bs = int(cfg.get("epochs", 2)), int(cfg.get("batch_size", 128))
    lr, wd = float(cfg.get("lr", 2e-5)), float(cfg.get("weight_decay", 0.0))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)   # always explicit: a silently
    eff_wd = opt.param_groups[0]["weight_decay"]                        # defaulted wd bit this project
    steps_per_ep = len(tr) // bs
    total = steps_per_ep * epochs
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
             if cfg.get("schedule", "constant") == "cosine" else None)
    if verbose:
        print(f"[train] {cfg.get('name','cfg')} head={cfg.get('head','H0')} prompts={cfg.get('prompts','T0')} "
              f"params={head.n_trainable():,} lr={lr:g} wd={eff_wd:g} bs={bs} ep={epochs} "
              f"steps={total:,} sched={cfg.get('schedule','constant')} cw={cfg.get('class_weight','none')} "
              f"train_data={arm} n_train={len(tr):,}", flush=True)

    best = {"fake_rec@real98": -1.0}
    best_state, gstep, t0 = None, 0, time.time()
    every = int(cfg.get("eval_every", 0)) or 10 ** 9
    for ep in range(epochs):
        g = torch.Generator(device=device).manual_seed(int(cfg.get("seed", 0)) * 1000 + ep)
        perm = tr[torch.randperm(len(tr), generator=g, device=device)]
        for s in range(steps_per_ep):
            idx = perm[s * bs:(s + 1) * bs]
            z = head(b["feats"][idx].float())
            loss = F.cross_entropy(z, y_all[idx], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if sched: sched.step()
            gstep += 1
            if gstep % every == 0 or gstep == total:
                m = evaluate_split(head, b, dev_which, level=levels[0])
                if verbose:
                    print(f"[train] ep{ep} step {gstep}/{total} loss {loss.item():.4f} "
                          f"| devA {M.fmt(m)}", flush=True)
                if m["fake_rec@real98"] > best["fake_rec@real98"]:
                    best = m; best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                    best["_gstep"] = gstep
    final = evaluate_split(head, b, dev_which, level=levels[0])
    if best["fake_rec@real98"] < 0 or final["fake_rec@real98"] >= best["fake_rec@real98"]:
        best, best_state = final, {k: v.detach().clone() for k, v in head.state_dict().items()}
        best["_gstep"] = gstep
    else:
        head.load_state_dict(best_state)
    # the same weights scored at every requested contamination level, so a ranking-stability test
    # downstream compares like with like instead of re-training per level
    best["_levels"] = {}
    for lv in levels:
        lm = evaluate_split(head, b, dev_which, level=lv)
        best["_levels"][lv] = {k: v for k, v in lm.items() if isinstance(v, (int, float))}
    best["_train_s"] = time.time() - t0
    best["_params"] = head.n_trainable()
    best["_scale"] = float(head.log_scale.detach().exp()) if hasattr(head, "log_scale") else None
    # Saturation is FLAGGED, not thrown, and the caller decides. Two different situations produce a
    # DEV bin_auc >= 0.9999 and they need different responses:
    #   * a leaked split (the failure this guard was written for: this project shipped a val
    #     auc=1.0000 once), and
    #   * a genuinely strong encoder on a clean split -- PE-G14 (1.88 B vision) is far stronger than
    #     the CLIP-L/336 that GSD scores 0.9913 with, so saturating a group-disjoint DEV split here is
    #     a plausible real outcome, not necessarily a bug.
    # Aborting on the second case would throw away the whole sweep. So: record it, say it loudly, and
    # let train_spc.main() (a single deliberate run) treat it as fatal while sweep.py keeps the numbers
    # and simply prefers non-saturated configs when it can.
    if best["bin_auc"] >= LEAK_AUC and cfg.get("leak_guard", True):
        best["_leak_suspect"] = True
        if verbose:
            print(f"[train] !! DEV bin_auc={best['bin_auc']:.6f} >= {LEAK_AUC} for "
                  f"{cfg.get('name','cfg')}: this metric cannot rank candidates. Either the split "
                  f"leaked or the features are that good -- fake_rec@real98 "
                  f"({best['fake_rec@real98']:.6f}) is the discriminator either way.", flush=True)
    if verbose:
        print(f"[train] DONE {best['_train_s']:.1f}s | devA {M.fmt(best)} "
              f"acc3={best.get('acc3_NOT-A-DECISION-METRIC',float('nan')):.4f}(NOT-A-DECISION-METRIC) "
              f"scale={best['_scale']}", flush=True)
    return head, best


def zeroshot(cfg, b, protos, dev_which=1, levels=("c99",)):
    head = make_head(cfg, protos, b["device"])
    m = evaluate_split(head, b, dev_which, level=levels[0])
    m["_levels"] = {lv: {k: v for k, v in evaluate_split(head, b, dev_which, level=lv).items()
                         if isinstance(v, (int, float))} for lv in levels}
    m["_params"] = head.n_trainable()
    return head, m


def class_means(b, device):
    """C1 control: prototypes = L2-normalised class means of the TRAIN features (no text at all)."""
    tr = b["split"] == 0
    out = []
    for c in range(3):
        m = tr & (b["labels"] == c)
        out.append(F.normalize(b["feats"][m].float().mean(0), dim=-1))
    return torch.stack(out).to(device)


def fingerprint(cache=os.path.join(ROOT, "cache/fingerprint.json")):
    """What the head was trained against. infer.py/evaluate.py assert these match at serve time --
    the project's most-repeated failure is a train/serve preprocessing skew that raises nothing and
    just scores worse (A2 crop/letterbox, GSD double-crop, SeLop RandomResizedCrop/Resize)."""
    ck = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"
    if os.path.exists(cache):
        fp = json.load(open(cache))
        # A CACHED fingerprint that is never re-verified is not a fingerprint -- if the encoder file is
        # replaced, every later checkpoint would inherit the stale hash and infer.py's check would then
        # "pass" against the wrong weights. Re-hash and refuse to reuse a stale entry.
        live_sha = cache_io.sha256(ck)
        if fp.get("encoder_sha256") != live_sha:
            raise SystemExit(
                f"[train] {cache} records encoder_sha256={str(fp.get('encoder_sha256'))[:16]}... but the "
                f"encoder on disk hashes to {live_sha[:16]}.... The cached features were computed with "
                f"one of them and this head would be calibrated against the other. Delete the cache "
                f"and re-extract, or restore the original encoder.")
        return fp
    sys.path.insert(0, os.path.join(ROOT, "perception_models"))
    import core.vision_encoder.transforms as pt
    fp = {"transform": repr(pt.get_image_transform(448)), "image_size": 448, "dtype": "bf16",
          "storage_dtype": "fp16", "normalize": True,
          "encoder_sha256": cache_io.sha256(ck), "encoder": os.path.basename(ck),
          "preprocess_note": "PE-native SQUASH resize + Normalize(0.5,0.5). NEVER paas/preprocess.py "
                             "letterbox -- different pixels, no error, plausible-looking score."}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(fp, open(cache, "w"), indent=1)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default="cache/train")
    ap.add_argument("--expect", default="train", choices=["train", "eval", "none"])
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    exp = {"train": cache_io.EXPECT_TRAIN, "eval": cache_io.EXPECT_EVAL, "none": None}[a.expect]
    b = load_bundle(a.cache, device=a.device, expect=exp)
    protos = torch.load(os.path.join(ROOT, "cache/prototypes.pt"))["protos"]
    protos = {k: v.to(a.device) for k, v in protos.items()}
    if cfg.get("prompts") == "C1":
        cfg["_class_means"] = class_means(b, a.device)
    head, m = train(cfg, b, protos)
    if m.get("_leak_suspect"):
        # Loud, recorded, and NOT fatal. The guard exists to stop a saturated metric being presented
        # silently as a clean result -- a flag carried into the checkpoint, the metrics JSON and the
        # report accomplishes exactly that, whereas aborting an automated chain here would destroy the
        # run without adding any information.
        print(f"[train] {'='*78}\n[train] SATURATED DEV METRIC: bin_auc={m['bin_auc']:.6f} >= "
              f"{LEAK_AUC}. This is recorded as `leak_suspect` in the checkpoint and the metrics JSON, "
              f"and it means bin_auc cannot rank anything on this split. Judge this run on "
              f"fake_rec@real98={m['fake_rec@real98']:.6f} and on the TESTSET numbers, not on DEV AUC."
              f"\n[train] {'='*78}", flush=True)
    # thresholds are fitted HERE, on DEV-A, and stored -- so testset/axon1 apply them unchanged.
    # A requested threshold that was not produced is an error, never a silent skip.
    tau = {}
    for t in (95, 98, 99):
        k = f"tau@real{t}"
        if k not in m or not np.isfinite(m[k]):
            raise SystemExit(f"[train] threshold {k} was requested but not produced -- refusing to "
                             f"save a checkpoint whose thresholds are missing.")
        tau[str(t)] = float(m[k])
    dupA = evaluate_split(head, b, 1, dedup=False, level="raw")
    print(f"[train] DEV-A raw, duplicate records kept: {M.fmt(dupA)} "
          f"(reported metric uses distinct, scoreable, decontaminated images)", flush=True)
    levels = {}
    for lv in ("raw", "c99", "c95"):
        lm = evaluate_split(head, b, 1, level=lv)
        levels[lv] = {k: v for k, v in lm.items() if isinstance(v, (int, float))}
        print(f"[train] DEV-A [{lv:<3}] n={lm['n']:>6,} {M.fmt(lm)}", flush=True)
    devB = evaluate_split(head, b, 2)
    print(f"[train] DEV-B (confirmation, touched once): {M.fmt(devB)}", flush=True)
    dd = devB["fake_rec@real98"] - m["fake_rec@real98"]
    print(f"[train] DEV-B - DEV-A fake_rec@real98 = {dd:+.6f} "
          f"{'OK' if dd >= -0.005 else '<-- WORSE THAN -0.005: selection is noise-dominated'}",
          flush=True)
    out = a.out or os.path.join(ROOT, "runs/spc", cfg.get("name", "run") + ".pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
                "state_dict": head.state_dict(), "devA": {k: v for k, v in m.items()},
                "devB": {k: v for k, v in devB.items()}, "tau": tau,
                "leak_suspect": bool(m.get("_leak_suspect")),
                "fingerprint": fingerprint(), "split_meta": b["split_meta"]}, out)
    # sibling JSON: the report builder must never have to open a torch checkpoint to quote a number
    side = out.replace(".pt", "_metrics.json")
    json.dump({"cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
               "params": head.n_trainable(),
               "devA": {k: v for k, v in m.items() if isinstance(v, (int, float, list))},
               "devB": {k: v for k, v in devB.items() if isinstance(v, (int, float, list))},
               "tau": tau, "ckpt": out,
               "leak_suspect": bool(m.get("_leak_suspect")),
               "devA_levels": levels,
               "devA_with_duplicate_records": {k: v for k, v in dupA.items()
                                               if isinstance(v, (int, float))},
               "n_distinct_cached_images": b.get("n_distinct")}, open(side, "w"), indent=1)
    print(f"[train] -> {out}\n[train] -> {side}\n[train] tau (DEV-A fitted): {tau}", flush=True)


if __name__ == "__main__":
    main()

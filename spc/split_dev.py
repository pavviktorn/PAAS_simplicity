"""Group-disjoint TRAIN / DEV-A / DEV-B split.

Why not a random split: consecutive frames of one clip are near-duplicates, so a random split puts
the same face in train and dev and every dev metric saturates -- exactly the failure already
recorded in this project (a 9-class branch logged val auc=1.0000 and saved it as `best`). Groups are
whole clips / directories, so a group is never on both sides.

The assignment is STATELESS: split(p) = f(blake2b(seed, groupkey(p))). Nothing is stored per image,
so the split cannot drift out of sync with the feature-cache row order (the cache is stored in
4-way strided shard order, NOT manifest order -- a stored index array would be a silent trap).

DEV-A selects everything (prompts, head, hyperparameters). DEV-B is touched ONCE, to confirm the
single winner. The testset selects nothing.
"""
from __future__ import annotations
import collections, hashlib, json, os, re, sys
import numpy as np

# Bump when groupkey(), FRAC, BIG_GROUP or the assignment rule changes. Derived artifacts
# (cache/dev_contam.npz) embed this, so a logic change invalidates them instead of silently
# re-using positional masks built under the old rule.
GROUPKEY_VERSION = 2

SEED = 1337
BIG_GROUP = 5000
FRAC = {"train": 0.92, "devA": 0.04, "devB": 0.04}
TRAIN, DEVA, DEVB = 0, 1, 2

# Strip exactly ONE trailing numeric token, not a greedy run of them.
#   greedy ([_-](f|frame)?\d+)+$ turned "05_03_05_070.jpg" into group key "05", merging entire
#   device/session trees into single 8,330-image "groups" -- an artefact that CREATED most of the
#   oversized groups. Measured on this manifest: greedy -> 464,105 groups, max 9,128, 9 oversized
#   covering 60,829 images; single-token -> 472,064 groups, max 6,990, 6 oversized covering 36,901.
_FRAME_TAIL = re.compile(r'[_-](f|frame)?\d+$')


def groupkey(p: str) -> str:
    d, f = os.path.split(p)
    s = os.path.splitext(f)[0]
    if s.isdigit():
        return d                                   # bare-integer frame names -> whole dir is a clip
    return d + "|" + _FRAME_TAIL.sub("", s)


def _u01(key: str, seed: int = SEED) -> float:
    h = hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def assign(paths, big_groups=frozenset(), seed: int = SEED) -> np.ndarray:
    """-> int8 array of TRAIN/DEVA/DEVB per path.

    Oversized groups go ENTIRELY TO TRAIN. The earlier version hashed the PATH for them so they would
    span all splits -- which silently broke group-disjointness, because those groups are not the flat
    photo pools the comment claimed: two are video frame dumps (T4_1_Video...), and per-image hashing
    scattered temporally adjacent frames of one clip across TRAIN and DEV-A. Measured consequence
    before this fix: ~1,531 of DEV-A's 54,914 rows had a temporally adjacent frame in TRAIN (2.8% of
    DEV-A, 4.4% of the fake_recall denominator), biasing fake recall and AUC upward -- invisibly,
    because DEV-B carried the same contamination so the DEV-B minus DEV-A check could not see it, and
    the leak guard only fires at bin_auc >= 0.9999.

    Forcing them to TRAIN instead of assigning the whole group randomly is deliberate: all 6 are
    SINGLE-CLASS FAKE and 5-7k images each, so a whole-group draw could drop 7,000 same-class rows
    into a 54,914-row DEV-A and skew it. Cost, stated plainly: DEV-A/DEV-B never see these 6 fake
    sources (36,901 images, 2.71% of the trainset). They are all fake, and DEV keeps ~34k fake rows
    from other groups, so the selection metric is essentially unaffected -- but selection does not
    cover those two video-monitor spoof types or those two flat GAN pools.
    """
    a = np.empty(len(paths), dtype=np.int8)
    t, d = FRAC["train"], FRAC["train"] + FRAC["devA"]
    for i, p in enumerate(paths):
        g = groupkey(p)
        if g in big_groups:
            a[i] = TRAIN
            continue
        u = _u01(g, seed)
        a[i] = TRAIN if u < t else (DEVA if u < d else DEVB)
    return a


def main():
    sys.path.insert(0, "/datasets/work/vLLM/temp/PAAS_ensemble_v4/gsd")
    from get_label import get_label_all, MAKEUP, PAD, UNKNOWN
    root = "/datasets/work/vLLM/temp/PAAS_simplicity"
    src = os.path.join(root, "manifests/train.json")
    recs = json.load(open(src))
    paths = [r["image"] for r in recs]
    labs = np.array([r["label"] for r in recs], dtype=np.int64)

    # group sizes + purity
    sizes, gl = {}, {}
    for p, l in zip(paths, labs):
        g = groupkey(p)
        sizes[g] = sizes.get(g, 0) + 1
        gl.setdefault(g, set()).add(int(l))
    impure = [g for g, s in gl.items() if len(s) > 1]
    big = {g for g, n in sizes.items() if n > BIG_GROUP}
    print(f"[split] {len(sizes):,} groups | median {np.median(list(sizes.values())):.0f} "
          f"max {max(sizes.values()):,} | multi-class groups {len(impure)} | "
          f"oversized(>{BIG_GROUP}) {len(big)} covering {sum(sizes[g] for g in big):,} imgs")
    if impure:
        raise SystemExit(f"[split] {len(impure)} groups span >1 class -> grouping is wrong, aborting")

    # composition of the oversized groups, so "forced to TRAIN" is auditable rather than asserted
    for g in sorted(big, key=lambda g: -sizes[g]):
        h = collections.Counter(int(l) for p, l in zip(paths, labs) if groupkey(p) == g)
        print(f"[split] oversized -> TRAIN: n={sizes[g]:>6,} labels={dict(h)}  ...{g[-70:]}")
    a = assign(paths, big)
    # the invariant the previous version violated
    bad = [g for g in big if any(a[i] != TRAIN for i, p in enumerate(paths) if groupkey(p) == g)]
    if bad:
        raise SystemExit(f"[split] {len(bad)} oversized groups leaked out of TRAIN -- aborting")
    dev_groups = {groupkey(p) for p, s_ in zip(paths, a) if s_ != TRAIN}
    if dev_groups & set(big):
        raise SystemExit("[split] an oversized group appears in DEV -- group-disjointness violated")
    info = {"seed": SEED, "big_group_thresh": BIG_GROUP, "frac": FRAC,
            "sha256_train_json": None, "n_groups": len(sizes), "n_big_groups": len(big),
            "big_groups": sorted(big), "counts": {}}
    for nm, k in (("train", TRAIN), ("devA", DEVA), ("devB", DEVB)):
        m = a == k
        h = {int(c): int(((labs == c) & m).sum()) for c in (0, 1, 2)}
        info["counts"][nm] = {"n": int(m.sum()), "hist": h}
        print(f"[split] {nm:<5} n={int(m.sum()):>9,}  real={h[0]:>7,} pad={h[1]:>7,} deepfake={h[2]:>7,}")
    import cache_io
    info["sha256_train_json"] = cache_io.sha256(src)
    out = os.path.join(root, "manifests/split.json")
    json.dump(info, open(out, "w"), indent=1)
    print(f"[split] -> {out}  sha256(train.json)={info['sha256_train_json'][:16]}...")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

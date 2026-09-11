"""Cache provenance: what a shard was built FROM, so it can never be silently reused for something else.

Both extractors used to skip an existing shard purely because the filename existed. Nothing recorded
the manifest, the encoder, the transform, or the shard count -- so changing any of them and re-running
would silently mix incompatible features. The concrete hazard is not hypothetical: re-running with
--n-divided 8 next to existing train_0..3 leaves four shards from the OLD striding and four from the
new, and every label lines up with the wrong feature row.

Provenance lives in a SIDECAR json next to each shard rather than inside the .npz. Adding a field to
the npz would mean rewriting 3.7 GB of already-verified output from a two-hour extraction, and a
rewrite can only lose data it cannot gain.
"""
from __future__ import annotations
import hashlib, json, os

FIELDS_MUST_MATCH = ("src", "src_sha256", "encoder_sha256", "transform", "image_size",
                     "dtype", "storage_dtype", "n_divided")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sidecar_path(out_npz):
    return out_npz[:-4] + ".meta.json" if out_npz.endswith(".npz") else out_npz + ".meta.json"


def build(src, encoder_ckpt, transform_repr, image_size, dtype, storage_dtype,
          n_divided, which_part, n_items_total):
    return {"src": os.path.abspath(src),
            "src_sha256": sha256_file(src) if os.path.isfile(src) else None,
            "encoder_sha256": sha256_file(encoder_ckpt),
            "transform": transform_repr, "image_size": int(image_size),
            "dtype": dtype, "storage_dtype": storage_dtype,
            "n_divided": int(n_divided), "which_part": int(which_part),
            "n_items_total": int(n_items_total)}


def write(out_npz, meta):
    json.dump(meta, open(sidecar_path(out_npz), "w"), indent=1)


def read(out_npz):
    p = sidecar_path(out_npz)
    return json.load(open(p)) if os.path.exists(p) else None


def check_reusable(out_npz, meta):
    """Called before skipping an existing shard. Returns (ok, reason).

    A shard with NO sidecar is refused, not trusted. The previous version returned ok=True with a
    "reusing on trust" note and the caller then WROTE the current provenance onto it -- which converted
    an unverified file into a certified one and destroyed the very evidence the check exists to find.
    Set ALLOW_LEGACY_CACHE=1 to accept an unstamped shard deliberately; even then nothing is stamped.
    """
    old = read(out_npz)
    if old is None:
        if os.environ.get("ALLOW_LEGACY_CACHE") == "1":
            return True, ("no provenance sidecar; ALLOW_LEGACY_CACHE=1 so reusing it explicitly "
                          "WITHOUT stamping (its true origin remains unknown)")
        return False, ("no provenance sidecar, so what this shard was built from is unknown. Either "
                       "delete it and re-extract, or run spc/stamp_provenance.py (which verifies what "
                       "it can before recording), or set ALLOW_LEGACY_CACHE=1 to accept it knowingly")
    diff = [k for k in FIELDS_MUST_MATCH if old.get(k) != meta.get(k)]
    if old.get("which_part") != meta.get("which_part"):
        diff.append("which_part")
    if diff:
        det = "; ".join(f"{k}: cached={str(old.get(k))[:48]!r} now={str(meta.get(k))[:48]!r}"
                        for k in diff)
        return False, f"provenance MISMATCH on {diff}: {det}"
    return True, "provenance matches"


def validate_set(prefix, files):
    """All shards of one cache must agree, and there must be exactly n_divided of them."""
    metas = [(f, read(f)) for f in files]
    have = [(f, m) for f, m in metas if m]
    if not have:
        return {"provenance": "absent for every shard (legacy cache)"}
    ref = have[0][1]
    for f, m in have[1:]:
        bad = [k for k in FIELDS_MUST_MATCH if m.get(k) != ref.get(k)]
        if bad:
            raise SystemExit(f"[prov] {os.path.basename(f)} disagrees with "
                             f"{os.path.basename(have[0][0])} on {bad} -- these shards were built from "
                             f"different inputs and must not be concatenated.")
    parts = sorted(m["which_part"] for _, m in have)
    nd = ref.get("n_divided")
    # The completeness check must run whenever ANY shard is stamped. Gating it on
    # len(have) == len(files) meant a PARTIALLY stamped set skipped it entirely -- exactly the state a
    # half-finished re-extraction leaves behind.
    if nd is not None:
        if len(files) != nd:
            raise SystemExit(f"[prov] cache {prefix} has {len(files)} shard file(s) but the provenance "
                             f"says n_divided={nd}: the set is incomplete or mixes two stridings, so "
                             f"rows and labels would mis-align.")
        if len(have) != len(files):
            raise SystemExit(f"[prov] cache {prefix} is PARTIALLY stamped ({len(have)}/{len(files)} "
                             f"shards have provenance). A half-stamped set is the signature of an "
                             f"interrupted re-extraction; refusing to guess which half is current.")
        if parts != list(range(nd)):
            raise SystemExit(f"[prov] cache {prefix} covers parts {parts}, expected 0..{nd-1}.")
    return {"provenance": "consistent", "n_shards_with_meta": len(have),
            "src": ref.get("src"), "src_sha256": ref.get("src_sha256"),
            "encoder_sha256": ref.get("encoder_sha256"), "n_divided": nd}

"""Build class prototypes from the FROZEN PE text tower. This is the whole idea of PE-SPC:
the text encoder is used ONCE, to place the prototypes, and then never again.

The paper is binary ("AI art" vs "a real photo"). A third class has no paper wording, so the 3-class
triplets below are candidates to be SELECTED ON DEV-A (never on the testset). Class order is always
(real, pad, deepfake).

Measured risk this addresses: cos("a real photo", "AI art") = 0.82 in this space. Text embeddings of
short prompts are crowded, so a badly-chosen triplet starts the three prototypes nearly collinear
and the calibration has to undo the initialisation instead of using it. Prototype separation is
printed for every candidate so a collapsed triplet is visible before training, not after.
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn.functional as F

ROOT = "/datasets/work/vLLM/temp/PAAS_simplicity"
sys.path.insert(0, os.path.join(ROOT, "perception_models"))
import core.vision_encoder.pe as pe                  # noqa: E402
import core.vision_encoder.transforms as pt          # noqa: E402

CKPT = "/datasets/work/vLLM/temp/PE-Core-G14-448/PE-Core-G14-448.pt"

# (real, pad, deepfake)
TRIPLETS = {
    "T0": ("a real photo", "a photo of a screen", "AI art"),                      # paper-anchored
    "T1": ("a real photo", "a spoof photo", "AI art"),
    "T2": ("a real face", "a presentation attack", "a deepfake face"),
    "T3": ("a photo of a real human face",
           "a photo of a face displayed on a screen or printed on paper",
           "a digitally manipulated or AI-generated face"),
    "T4": ("a live face captured directly by a camera",
           "a recaptured face: a photograph of a screen, print, or mask",
           "a synthetically generated or face-swapped image"),
    "T5": ("a genuine photograph of a person",
           "a photo of a printed photo, a phone screen, or a silicone mask",
           "a face swap or GAN-generated face"),
    "T6": ("real", "replay", "deepfake"),
    "T7": ("a real photo", "a photo of a photo", "AI art"),
    "T8": ("a real photo", "a photo of a face on a screen or a mask", "AI art"),
    "T9": ("a bona fide face presentation", "a presentation attack instrument", "a manipulated face"),
}

# K=4 variants per class for the multi-prototype head (H3): PAD is 7 visually distinct attack
# families, so one prototype per class is a real bottleneck by construction.
K4 = {
    "real": ["a real photo", "a live face captured directly by a camera",
             "a genuine photograph of a person", "a bona fide face presentation"],
    "pad": ["a photo of a face displayed on a screen", "a photo of a printed face on paper",
            "a face wearing a silicone or latex mask", "a photo of a face on a cloth or textile print"],
    "deepfake": ["AI art", "a face swap", "a GAN-generated face",
                 "a digitally manipulated face"],
}


def main():
    dev = os.environ.get("PROTO_DEVICE", "cpu")
    print(f"[proto] loading PE on {dev} (text tower only is used, but from_config builds both)", flush=True)
    model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True, checkpoint_path=CKPT)
    model = model.to(dev).eval()
    # context_length 72 for this model -- the module default DEFAULT_CONTEXT_LENGTH=77 is WRONG here
    tok = pt.get_text_tokenizer(model.context_length)
    print(f"[proto] context_length={model.context_length} clip_dim={model.text_projection.shape[-1]}"
          f" logit_scale.exp()={model.logit_scale.exp().item():.4f}", flush=True)

    @torch.no_grad()
    def emb(strs):
        t = tok(list(strs)).to(dev)
        return model.encode_text(t, normalize=True).float()

    out = {}
    for name, tri in TRIPLETS.items():
        e = emb(tri)                                            # (3,1280) unit
        out[name] = e
        g = e @ e.t()
        print(f"[proto] {name:<3} cos(real,pad)={g[0,1]:.4f} cos(real,df)={g[0,2]:.4f} "
              f"cos(pad,df)={g[1,2]:.4f}  mean_offdiag={((g.sum()-3)/6):.4f}   {tri[0]!r}/{tri[1]!r}/{tri[2]!r}",
              flush=True)

    # T10 = prompt ensemble: mean of the L2-normalised embeddings of T0..T9 per class, renormalised
    stack = torch.stack([out[k] for k in TRIPLETS])              # (10,3,1280)
    out["T10"] = F.normalize(stack.mean(0), dim=-1)
    g = out["T10"] @ out["T10"].t()
    print(f"[proto] T10 cos(real,pad)={g[0,1]:.4f} cos(real,df)={g[0,2]:.4f} cos(pad,df)={g[1,2]:.4f}"
          f"  (ensemble of T0-T9)", flush=True)

    # K=4 per class, class-major row order: real x4, pad x4, deepfake x4
    out["K4"] = torch.cat([emb(K4[c]) for c in ("real", "pad", "deepfake")], 0)
    print(f"[proto] K4 {tuple(out['K4'].shape)} (class-major: real0-3, pad4-7, deepfake8-11)")

    # Fingerprint the tower these prototypes came out of. Prototypes are only meaningful in the same
    # embedding space as the cached image features, and nothing in a 3x1280 tensor says which encoder
    # produced it.
    import hashlib as _h
    _s = _h.sha256()
    with open(CKPT, "rb") as _f:
        for _c in iter(lambda: _f.read(1 << 22), b""):
            _s.update(_c)
    fingerprint = {"encoder_sha256": _s.hexdigest(),
                   "context_length": int(model.context_length),
                   "clip_dim": int(model.text_projection.shape[-1]),
                   "logit_scale_exp": float(model.logit_scale.detach().exp()),
                   "tokenizer": "pt.get_text_tokenizer(model.context_length)"}
    dst = os.path.join(ROOT, "cache/prototypes.pt")
    torch.save({"protos": out, "triplets": TRIPLETS, "k4": K4, "fingerprint": fingerprint,
                "logit_scale_exp": float(model.logit_scale.detach().exp())}, dst)
    print(f"[proto] fingerprint encoder_sha256={fingerprint['encoder_sha256'][:16]}... "
          f"context_length={fingerprint['context_length']}", flush=True)
    print(f"[proto] -> {dst}", flush=True)


if __name__ == "__main__":
    main()

"""The trainable part of PE-SPC: C x 1280 prototypes. Nothing else trains.

Class order is FIXED PROJECT-WIDE as 0=real, 1=pad, 2=deepfake -- the OPPOSITE of the paper's
binary (0=generated, 1=real). The paper's bias initialisation ("1 for the generated class, 0 for
real") therefore becomes [0, 1, 1] here, NOT [1, 0, 0]. Getting that backwards trains fine, logs
fine, and just starts the optimiser on the wrong side of the boundary.

Topologies (H0 is the paper):
  H0 linear        logits = s*(x @ P^T) + b                       3*1280 + 3 = 3,843 params
  H1 +scale        H0 with s learnable (init 1.0 == exact H0 at step 0)          +1
  H2 cosine        prototypes re-normalised every forward
  H3 multi-proto   K prototypes per class, logits_c = logsumexp_k s*(x . p_ck)   K*3*1280 + 3
  H4 mlp           1280 -> 512 -> 3, DIAGNOSTIC ONLY: the ceiling of the frozen features
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

BIAS_INIT = (0.0, 1.0, 1.0)          # index 0 == real


class SPCHead(nn.Module):
    def __init__(self, dim=1280, n_classes=3, k=1, cosine=False,
                 learn_scale=True, scale_init=1.0, bias_init=BIAS_INIT):
        super().__init__()
        self.n_classes, self.k, self.cosine = n_classes, k, cosine
        self.proto = nn.Parameter(torch.zeros(n_classes * k, dim))
        self.bias = nn.Parameter(torch.tensor(list(bias_init), dtype=torch.float32))
        self.learn_scale = learn_scale
        # log-parameterised so the scale can never go negative and s=1 reproduces the paper exactly
        ls = torch.tensor(float(torch.log(torch.tensor(float(scale_init)))))
        self.log_scale = nn.Parameter(ls) if learn_scale else nn.Parameter(ls, requires_grad=False)

    @torch.no_grad()
    def init_prototypes(self, protos: torch.Tensor):
        """protos: (n_classes*k, dim), already L2-normalised. Row order = class-major."""
        assert protos.shape == self.proto.shape, f"{protos.shape} != {self.proto.shape}"
        self.proto.copy_(protos.to(self.proto.dtype))

    def forward(self, x):
        p = F.normalize(self.proto, dim=-1) if self.cosine else self.proto
        z = x @ p.t()                                        # (N, C*K)
        s = self.log_scale.exp()
        z = z * s
        if self.k > 1:
            z = torch.logsumexp(z.view(-1, self.n_classes, self.k), dim=-1)
        return z + self.bias

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPHead(nn.Module):
    """H4 -- diagnostic only, NOT deployable. Answers 'is the head or the encoder the bottleneck?'"""
    def __init__(self, dim=1280, n_classes=3, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_head(cfg):
    if cfg.get("head", "H0") == "H4":
        return MLPHead(cfg.get("dim", 1280), 3, cfg.get("hidden", 512))
    return SPCHead(dim=cfg.get("dim", 1280), n_classes=3, k=cfg.get("k", 1),
                   cosine=bool(cfg.get("cosine", False)),
                   learn_scale=bool(cfg.get("learn_scale", False)),
                   scale_init=float(cfg.get("scale_init", 1.0)),
                   bias_init=tuple(cfg.get("bias_init", BIAS_INIT)))

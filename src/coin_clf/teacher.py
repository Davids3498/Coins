"""v6 teacher architecture (copied from emp_model_v6.ipynb via emp_model_knowledge_distilation.ipynb).

Used only to load the frozen v6 ensemble/sub-classifier checkpoints and precompute
teacher soft labels for distillation — not used at serving time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceClassifier(nn.Module):
    def __init__(self, in_dim, n_cls, scale=20.0, margin=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_cls, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale; self.margin = margin

    def forward(self, x, labels=None):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        cos = x @ w.T
        if labels is None or not self.training:
            return self.scale * cos
        theta = cos.clamp(-1 + 1e-7, 1 - 1e-7).acos()
        target_cos = (theta + self.margin).cos()
        one_hot = torch.zeros_like(cos).scatter_(1, labels.unsqueeze(1), 1.0)
        return self.scale * (one_hot * target_cos + (1 - one_hot) * cos)


class StreamProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x): return self.proj(x)


class CoinHeadV6(nn.Module):
    def __init__(self, n_cls, full_dim=3840, portrait_dim=2560, legend_dim=2560,
                 proj_full=1024, proj_crop=512, hidden=2048, dropout=0.3):
        super().__init__()
        self.proj_full     = StreamProjector(full_dim,     proj_full, dropout=0.1)
        self.proj_portrait = StreamProjector(portrait_dim, proj_crop, dropout=0.1)
        self.proj_legend   = StreamProjector(legend_dim,   proj_crop, dropout=0.1)
        fused = proj_full + proj_crop + proj_crop
        self.body = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.cls = ArcFaceClassifier(hidden // 2, n_cls)

    def forward(self, x_full, x_portrait, x_legend, labels=None):
        fused = torch.cat([
            self.proj_full(x_full),
            self.proj_portrait(x_portrait),
            self.proj_legend(x_legend),
        ], dim=-1)
        return self.cls(self.body(fused), labels)


class SubHead(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=512, dropout=0.2, margin=0.4, scale=25.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.cls = ArcFaceClassifier(hidden // 2, n_cls, scale=scale, margin=margin)

    def forward(self, x, labels=None):
        return self.cls(self.body(x), labels)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    ArcFace head:
    - forward(x, label=None): returns scaled cosine logits
    - if label is provided: applies additive angular margin to target class
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        # normalize features & weights
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)

        cosine = F.linear(x_norm, w_norm).clamp(-1.0, 1.0)  # (B, out)
        if label is None:
            return cosine * self.s

        # theta + m
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits
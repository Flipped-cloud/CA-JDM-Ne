import torch
import torch.nn as nn
import torch.nn.functional as F


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)


class Discriminator_x(nn.Module):
    """
    Image discriminator.
    Returns:
      feat: (B, out_channels)
      s_x : (B, 1) if with_sx else None
    """
    def __init__(self, img_size=64, out_channels=512, SN_eps=1e-12, wide=True, with_sx=True, early_xfeat=False):
        super().__init__()
        self.SN_eps = SN_eps
        self.with_sx = with_sx

        ch = 64
        # simple SN conv stack with downsampling
        self.net = nn.Sequential(
            snconv2d(in_channels=3, out_channels=ch, kernel_size=3, stride=1, padding=1, eps=self.SN_eps),
            nn.LeakyReLU(0.2, inplace=True),

            snconv2d(in_channels=ch, out_channels=ch * 2, kernel_size=4, stride=2, padding=1, eps=self.SN_eps),
            nn.LeakyReLU(0.2, inplace=True),

            snconv2d(in_channels=ch * 2, out_channels=ch * 4, kernel_size=4, stride=2, padding=1, eps=self.SN_eps),
            nn.LeakyReLU(0.2, inplace=True),

            snconv2d(in_channels=ch * 4, out_channels=ch * 8, kernel_size=4, stride=2, padding=1, eps=self.SN_eps),
            nn.LeakyReLU(0.2, inplace=True),

            snconv2d(in_channels=ch * 8, out_channels=out_channels, kernel_size=3, stride=1, padding=1, eps=self.SN_eps),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)

    def forward(self, x):
        h = self.net(x)
        feat = self.pool(h).flatten(1)  # (B, out_channels)
        s_x = self.head(feat) if self.with_sx else None
        return feat, s_x


class Discriminator_lbl(nn.Module):
    """
    Emotion(label) discriminator.
    Input:
      - z can be LongTensor (B,) for hard label indices
      - or FloatTensor (B,C) for soft distribution / onehot
    Returns:
      feat: (B, out_channels)
      s  : (B, 1)
    """
    def __init__(self, in_channels=7, out_channels=512, SN_eps=1e-12, use_embedding=True):
        super().__init__()
        self.SN_eps = SN_eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.embedding = sn_embedding(num_embeddings=in_channels, embedding_dim=out_channels, eps=self.SN_eps)

        self.mlp = nn.Sequential(
            snlinear(in_features=out_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
            snlinear(in_features=out_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
        )
        self.head = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)

    def forward(self, z):
        if z.dtype in (torch.int32, torch.int64):
            # hard labels
            feat = self.embedding(z)
        else:
            # soft labels / onehot: (B,C) -> weighted sum of embedding vectors
            # embedding.weight: (C, out_channels)
            w = self.embedding.weight
            if w.device != z.device:
                w = w.to(z.device)
            feat = torch.matmul(z, w)

        feat = self.mlp(feat)
        s = self.head(feat)
        return feat, s


class Discriminator_lmk(nn.Module):
    """Landmark discriminator (MLP). Input: (B,136)"""
    def __init__(self, in_channels=136, out_channels=512, SN_eps=1e-12):
        super().__init__()
        self.SN_eps = SN_eps
        self.mlp = nn.Sequential(
            snlinear(in_features=in_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
            snlinear(in_features=out_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
        )
        self.head = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)

    def forward(self, lmk):
        feat = self.mlp(lmk)
        s = self.head(feat)
        return feat, s


class Discriminator_z0(nn.Module):
    """z0 discriminator (MLP). Input: (B, noise_dim)"""
    def __init__(self, in_channels=100, out_channels=512, SN_eps=1e-12):
        super().__init__()
        self.SN_eps = SN_eps
        self.mlp = nn.Sequential(
            snlinear(in_features=in_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
            snlinear(in_features=out_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
        )
        self.head = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)

    def forward(self, z0):
        feat = self.mlp(z0)
        s = self.head(feat)
        return feat, s


class Discriminator_joint(nn.Module):
    """
    Joint discriminator over concatenated modality features.
    Input feats are already encoded vectors (B,512).
    Output: s_joint (B,1)
    """
    def __init__(self, in_channels=512 * 3, out_channels=1024, SN_eps=1e-12):
        super().__init__()
        self.SN_eps = SN_eps
        self.mlp = nn.Sequential(
            snlinear(in_features=in_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
            snlinear(in_features=out_channels, out_features=out_channels, eps=self.SN_eps),
            nn.ReLU(inplace=True),
        )
        self.head = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)

    def forward(self, *feats):
        h = torch.cat(list(feats), dim=1)
        h = self.mlp(h)
        return self.head(h)
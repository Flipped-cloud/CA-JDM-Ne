import torch
import torch.nn as nn
from .attention import CCAM, SCAM

class Encoder(nn.Module):
    def __init__(self, img_size=224, fc_layer=512, latent_dim=10, noise_dim=100):
        super(Encoder, self).__init__()
        self.fc_layer = fc_layer
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        # --- 修改点：适配 224x224 输入 ---
        if img_size == 256: self.final_size = 8
        elif img_size == 224: self.final_size = 7  # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        elif img_size == 128: self.final_size = 4
        elif img_size == 96: self.final_size = 3
        elif img_size == 64: self.final_size = 2
        elif img_size == 32: self.final_size = 1
        else: raise ValueError(f"Unsupported img_size: {img_size}")

        # =====================================================================
        # Stage 1 (Input -> /2)
        # =====================================================================
        self.conv_block1_f = self._make_block(3, 64)   # Face Stream
        self.conv_block1_l = self._make_block(3, 64)   # Landmark Stream
        self.ccam1 = CCAM(64)
        self.scam1 = SCAM(64)

        # =====================================================================
        # Stage 2 (/2 -> /4)
        # =====================================================================
        self.conv_block2_f = self._make_block(64, 128)
        self.conv_block2_l = self._make_block(64, 128)
        self.ccam2 = CCAM(128)
        self.scam2 = SCAM(128)

        # =====================================================================
        # Stage 3 (/4 -> /8)
        # =====================================================================
        self.conv_block3_f = self._make_block(128, 256, layers=3)
        self.conv_block3_l = self._make_block(128, 256, layers=3)
        self.ccam3 = CCAM(256)
        self.scam3 = SCAM(256)

        # =====================================================================
        # Stage 4 (/8 -> /16)
        # =====================================================================
        self.conv_block4_f = self._make_block(256, 512, layers=3)
        self.conv_block4_l = self._make_block(256, 512, layers=3)
        self.ccam4 = CCAM(512)
        self.scam4 = SCAM(512)

        # =====================================================================
        # Stage 5 (/16 -> /32)
        # =====================================================================
        self.conv_block5_f = self._make_block(512, 512, layers=3)
        self.conv_block5_l = self._make_block(512, 512, layers=3)
        self.ccam5 = CCAM(512)
        self.scam5 = SCAM(512)

        # =====================================================================
        # Face Head (Emotion & Latent Codes)
        # =====================================================================
        self.embedding = nn.Sequential(
            nn.Linear(512 * self.final_size * self.final_size, self.fc_layer),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.fc_layer, self.fc_layer),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.mean_layer = nn.Linear(self.fc_layer, self.noise_dim)
        self.logvar_layer = nn.Linear(self.fc_layer, self.noise_dim)
        self.c_layer = nn.Linear(self.fc_layer, self.latent_dim)

        # =====================================================================
        # Landmark Head (New Output)
        # =====================================================================
        self.landmark_layer = nn.Sequential(
            nn.Linear(512 * self.final_size * self.final_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 136)
        )

        self._init_weights()

    def _make_block(self, in_channels, out_channels, layers=2):
        """Helper to create VGG-style blocks"""
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        block.append(nn.ReLU(inplace=True))
        
        for _ in range(layers - 1):
            block.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            block.append(nn.ReLU(inplace=True))
            
        block.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))
        return nn.Sequential(*block)

    def _init_weights(self):
        # He init for conv/linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def sample_z(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        return z_mu + z_std * eps

    def forward(self, x):
        # ---------------- Stage 1 ----------------
        f1 = self.conv_block1_f(x)
        l1 = self.conv_block1_l(x)
        f1, l1 = self.ccam1(f1, l1)
        f1, l1 = self.scam1(f1, l1)

        # ---------------- Stage 2 ----------------
        f2 = self.conv_block2_f(f1)
        l2 = self.conv_block2_l(l1)
        f2, l2 = self.ccam2(f2, l2)
        f2, l2 = self.scam2(f2, l2)

        # ---------------- Stage 3 ----------------
        f3 = self.conv_block3_f(f2)
        l3 = self.conv_block3_l(l2)
        f3, l3 = self.ccam3(f3, l3)
        f3, l3 = self.scam3(f3, l3)

        # ---------------- Stage 4 ----------------
        f4 = self.conv_block4_f(f3)
        l4 = self.conv_block4_l(l3)
        f4, l4 = self.ccam4(f4, l4)
        f4, l4 = self.scam4(f4, l4)

        # ---------------- Stage 5 ----------------
        f5 = self.conv_block5_f(f4)
        l5 = self.conv_block5_l(l4)
        f5, l5 = self.ccam5(f5, l5)
        f5, l5 = self.scam5(f5, l5)

        # ---------------- Flatten ----------------
        f_flat = f5.view(f5.shape[0], -1)
        l_flat = l5.view(l5.shape[0], -1)

        # ---------------- Output Heads ----------------
        embedding = self.embedding(f_flat)
        mean = self.mean_layer(embedding)
        logvar = self.logvar_layer(embedding)
        z0 = self.sample_z(mean, logvar)          # noise vector
        z1 = self.c_layer(embedding)              # emotion logits
        pred_landmarks = self.landmark_layer(l_flat)  # [B,136]

        return mean, logvar, z0, z1, pred_landmarks
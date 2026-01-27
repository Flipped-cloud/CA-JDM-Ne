import functools
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math
from model.attention import CCAM, SCAM  # 导入协同注意力模块

class Encoder(nn.Module):
    def __init__(self, img_size=256, fc_layer=512, latent_dim=10, noise_dim=100):
        super(Encoder, self).__init__()
        self.fc_layer = fc_layer
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        # 计算最终的特征图大小
        if img_size == 256: self.final_size = 8
        elif img_size == 128: self.final_size = 4
        elif img_size == 96: self.final_size = 3
        elif img_size == 64: self.final_size = 2
        elif img_size == 32: self.final_size = 1
        else: raise ValueError("Unsupported img_size")

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
            nn.Dropout2d(0.5),
            nn.Linear(self.fc_layer, self.fc_layer),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )
        self.mean_layer = nn.Linear(self.fc_layer, self.noise_dim)
        self.logvar_layer = nn.Linear(self.fc_layer, self.noise_dim)
        self.c_layer = nn.Linear(self.fc_layer, self.latent_dim)

        # =====================================================================
        # Landmark Head (New Output)
        # =====================================================================
        # 输入: Landmark Stream 的 Flatten 特征
        # 输出: 68个点 * 2坐标 = 136
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

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
        # 1. Emotion & Latent Codes (From Face Stream)
        embedding = self.embedding(f_flat)
        mean = self.mean_layer(embedding)
        logvar = self.logvar_layer(embedding)
        z0 = self.sample_z(mean, logvar) # Noise vector
        z1 = self.c_layer(embedding)     # Predicted Emotion Label

        # 2. Landmarks (From Landmark Stream)
        pred_landmarks = self.landmark_layer(l_flat) # [B, 136]

        # 返回5个值，比原来多了一个 pred_landmarks
        return mean, logvar, z0, z1, pred_landmarks

    # 保留此函数以防原来的代码在某处调用它，但双流结构下加载单流VGG权重可能需要手动匹配
    def init_vggface_params(self, pretrained_model_path):
        print("Warning: Loading pretrained VGG weights into Dual-Stream Encoder is complex.")
        print("Initializing weights from scratch or skipping this step is recommended for now.")
        # 如果必须加载，需要编写复杂的 key mapping 逻辑 (conv_block1 -> conv_block1_f & conv_block1_l)
        pass
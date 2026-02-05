import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CCAM, SCAM, CBAM, SpatialAttention, ChannelAttention, CoordAtt


# ============================================================================
# IR-50 (ResNet-50 with ArcFace Structure) Building Blocks
# ============================================================================

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(nn.Module):
    """Improved Residual Bottleneck (for IR-50)"""
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(nn.Module):
    """Improved Residual Bottleneck with SE (for IR-SE-50)"""
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


# ============================================================================
# IR-50 Encoder with Attention Mechanisms
# ============================================================================

class IR50_Encoder(nn.Module):
    """
    IR-50 (Improved ResNet-50) Encoder for FER with Attention
    
    Architecture:
    - Input: (B, 3, 112, 112) or (B, 3, 224, 224)
    - IR-50 backbone with 4 stages
    - CCAM & SCAM attention modules at Stage 3 and Stage 4
    - Output: features (B, 2048), feature_maps [stage3, stage4]
    
    Reference: https://github.com/TreB1eN/InsightFace_Pytorch
    """
    def __init__(self, *args, fc_layer=512, **kwargs):
        super(IR50_Encoder, self).__init__()
        self.img_size = kwargs.get('img_size', 112)
        self.fc_layer = fc_layer
        self.latent_dim = kwargs.get('latent_dim', 10)
        self.noise_dim = kwargs.get('noise_dim', 100)
        self.use_dual_stream = kwargs.get('use_dual_stream', True)  # True: 双流架构, False: 单流+CBAM
        self.use_ca = kwargs.get('use_ca', False)
        
        # Choose block type
        if kwargs.get('use_se', False):
            block = BottleneckIRSE
        else:
            block = BottleneckIR
        
        # IR-50 configuration: [3, 4, 14, 3] blocks per stage
        num_blocks = [3, 4, 14, 3]
        
        # =====================================================================
        # Input Layer
        # =====================================================================
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
        # =====================================================================
        # Stage 1: 64 -> 64, output: (B, 64, 56, 56) for 112x112 input
        # =====================================================================
        modules_stage1 = []
        for _ in range(num_blocks[0]):
            modules_stage1.append(block(64, 64, 2 if _ == 0 else 1))
        self.stage1 = nn.Sequential(*modules_stage1)
        
        # =====================================================================
        # Stage 2: 64 -> 128, output: (B, 128, 28, 28)
        # =====================================================================
        modules_stage2 = []
        modules_stage2.append(block(64, 128, 2))
        for _ in range(1, num_blocks[1]):
            modules_stage2.append(block(128, 128, 1))
        self.stage2 = nn.Sequential(*modules_stage2)
        
        # =====================================================================
        # Stage 3: 128 -> 256, output: (B, 256, 14, 14)
        # Add CCAM & SCAM attention here
        # =====================================================================
        modules_stage3_f = []  # Face stream
        modules_stage3_l = []  # Landmark stream
        
        modules_stage3_f.append(block(128, 256, 2))
        modules_stage3_l.append(block(128, 256, 2))
        for _ in range(1, num_blocks[2]):
            modules_stage3_f.append(block(256, 256, 1))
            modules_stage3_l.append(block(256, 256, 1))
            
        self.stage3_f = nn.Sequential(*modules_stage3_f)
        self.stage3_l = nn.Sequential(*modules_stage3_l)
        self.ccam3 = CCAM(256, e_ratio=0.2)
        self.scam3 = SCAM(256, kernel_size=7)
        
        # CBAM attention for single-stream mode (Module 2 requirement)
        self.cbam3 = CBAM(256, reduction=16, kernel_size=7)
        # Coordinate Attention (CA) for single-stream mode
        self.ca3 = CoordAtt(256, reduction=32)
        
        # =====================================================================
        # Stage 4: 256 -> 512, output: (B, 512, 7, 7)
        # Add CCAM & SCAM attention here
        # =====================================================================
        modules_stage4_f = []
        modules_stage4_l = []
        
        modules_stage4_f.append(block(256, 512, 2))
        modules_stage4_l.append(block(256, 512, 2))
        for _ in range(1, num_blocks[3]):
            modules_stage4_f.append(block(512, 512, 1))
            modules_stage4_l.append(block(512, 512, 1))
            
        self.stage4_f = nn.Sequential(*modules_stage4_f)
        self.stage4_l = nn.Sequential(*modules_stage4_l)
        self.ccam4 = CCAM(512, e_ratio=0.2)
        self.scam4 = SCAM(512, kernel_size=7)
        
        # CBAM attention for single-stream mode (Module 2 requirement)
        self.cbam4 = CBAM(512, reduction=16, kernel_size=7)
        # Coordinate Attention (CA) for single-stream mode
        self.ca4 = CoordAtt(512, reduction=32)

        # Coordinate Attention (CA) for dual-stream mode (face/landmark)
        self.ca3_f = CoordAtt(256, reduction=32)
        self.ca3_l = CoordAtt(256, reduction=32)
        self.ca4_f = CoordAtt(512, reduction=32)
        self.ca4_l = CoordAtt(512, reduction=32)
        
        # =====================================================================
        # Output Layer: BN -> Dropout -> Flatten -> FC -> BN
        # =====================================================================
        if self.img_size == 112:
            final_spatial = 7  # 112 / 2^4 = 7
        elif self.img_size == 224:
            final_spatial = 14  # 224 / 2^4 = 14
        else:
            raise ValueError(f"Unsupported img_size: {self.img_size}. Use 112 or 224.")
        
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * final_spatial * final_spatial, 2048),
            nn.BatchNorm1d(2048)
        )
        
        # =====================================================================
        # Face Head (Emotion & Latent Codes) - for GAN training
        # =====================================================================
        # embedding 输入固定为 output_layer 的输出维度 2048
        self.embedding = nn.Sequential(
            nn.Linear(2048, fc_layer),
            nn.BatchNorm1d(fc_layer),
            nn.Dropout(p=0.4),
        )
        self.mean_layer = nn.Linear(self.fc_layer, self.noise_dim)
        self.logvar_layer = nn.Linear(self.fc_layer, self.noise_dim)
        self.c_layer = nn.Linear(self.fc_layer, self.latent_dim)
        
        # =====================================================================
        # Landmark Head (for Multi-task Learning)
        # =====================================================================
        self.landmark_layer = nn.Sequential(
            nn.Linear(512 * final_spatial * final_spatial, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 136)  # 68 landmarks * 2 (x, y)
        )
        
        # 确保存在自适应池化（embedding wrapper 也会做池化，但保留以便其他部分使用）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 期望的中间向量维度（来自 ArcFace/IR50 的通道数）
        self.output_dim = 2048

        # 使用兼容型 EmbeddingWrapper 替代直接的 Linear，
        # 可以接受 (B, C, H, W) 或 (B, N) 两种输入形式
        self.embedding = EmbeddingWrapper(target_dim=self.output_dim, fc_out=fc_layer, dropout=0.4)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, path):
        """
        Load pretrained weights from MS-Celeb-1M or other face recognition datasets
        
        Args:
            path: Path to the pretrained .pth file
        """
        if path is None or path == '':
            print("[WARNING] No pretrained weights path provided. Training from scratch.")
            return
        
        try:
            state_dict = torch.load(path, map_location='cpu')

            # Handle different state_dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Remove common prefixes
            cleaned_state = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                cleaned_state[k] = v

            # Map ArcFace IR-50 "body" blocks to our stage modules
            mapped_state = {}
            skipped = 0
            for k, v in cleaned_state.items():
                k_lower = k.lower()

                # Skip final FC/output layers (dimension mismatch)
                if 'fc' in k_lower or 'output_layer' in k:
                    skipped += 1
                    continue

                # Directly load input layer (names match)
                if k.startswith('input_layer.'):
                    mapped_state[k] = v
                    continue

                if k.startswith('body.'):
                    parts = k.split('.')
                    try:
                        block_idx = int(parts[1])
                    except (ValueError, IndexError):
                        skipped += 1
                        continue
                    rest = '.'.join(parts[2:])

                    # IR-50 block layout: [3, 4, 14, 3]
                    if block_idx <= 2:
                        mapped_state[f"stage1.{block_idx}.{rest}"] = v
                    elif block_idx <= 6:
                        mapped_state[f"stage2.{block_idx - 3}.{rest}"] = v
                    elif block_idx <= 20:
                        stage3_idx = block_idx - 7
                        mapped_state[f"stage3_f.{stage3_idx}.{rest}"] = v
                        mapped_state[f"stage3_l.{stage3_idx}.{rest}"] = v
                    elif block_idx <= 23:
                        stage4_idx = block_idx - 21
                        mapped_state[f"stage4_f.{stage4_idx}.{rest}"] = v
                        mapped_state[f"stage4_l.{stage4_idx}.{rest}"] = v
                    else:
                        skipped += 1
                    continue

                # Ignore any other unmatched keys
                skipped += 1

            # Load with strict=False to allow missing keys
            load_result = self.load_state_dict(mapped_state, strict=False)
            missing = len(load_result.missing_keys)
            unexpected = len(load_result.unexpected_keys)

            print(f"[INFO] Successfully loaded pretrained weights from {path}")
            print(f"[INFO] Mapped {len(mapped_state)} tensors. Skipped {skipped} tensors.")
            print(f"[INFO] Missing keys: {missing}, unexpected keys: {unexpected}.")
            
        except FileNotFoundError:
            print(f"[ERROR] Pretrained weights not found at {path}")
            print("[WARNING] Training from scratch.")
        except Exception as e:
            print(f"[ERROR] Failed to load pretrained weights: {e}")
            print("[WARNING] Training from scratch.")

    def sample_z(self, z_mu, z_logvar):
        """Reparameterization trick for VAE"""
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        return z_mu + z_std * eps

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            mean: Mean of latent distribution (B, noise_dim)
            logvar: Log variance of latent distribution (B, noise_dim)
            z0: Sampled noise vector (B, noise_dim)
            z1: Emotion logits (B, latent_dim)
            pred_landmarks: Predicted landmarks (B, 136)
        """
        # Input layer
        x = self.input_layer(x)
        
        # Stage 1 & 2: Shared backbone
        x = self.stage1(x)
        x = self.stage2(x)
        
        if self.use_dual_stream:
            # ===== 双流模式 (Co-Attention) =====
            # Stage 3: Split into two streams with co-attention
            f3 = self.stage3_f(x)
            l3 = self.stage3_l(x)
            if self.use_ca:
                f3 = self.ca3_f(f3)
                l3 = self.ca3_l(l3)
            f3, l3 = self.ccam3(f3, l3)
            f3, l3 = self.scam3(f3, l3)
            
            # Stage 4: Continue two streams with co-attention
            f4 = self.stage4_f(f3)
            l4 = self.stage4_l(l3)
            if self.use_ca:
                f4 = self.ca4_f(f4)
                l4 = self.ca4_l(l4)
            f4, l4 = self.ccam4(f4, l4)
            f4, l4 = self.scam4(f4, l4)
            
            # Store feature maps for potential use (e.g., visualization)
            feature_maps = [f3, f4, l3, l4]
            
            # Face stream -> use output_layer to produce 2048-d vector then embedding
            f_out = self.output_layer(f4)      # (B, 2048)
            embedding = self.embedding(f_out)  # (B, fc_layer)
            
            # Landmark stream -> Landmark predictions
            l_flat = l4.view(l4.shape[0], -1)
            pred_landmarks = self.landmark_layer(l_flat)
            
        else:
            # ===== 单流模式 (CBAM with Residual Connection) - Module 2 =====
            # Stage 3: Single stream with CBAM attention (residual connection)
            feat3 = self.stage3_f(x)
            if self.use_ca:
                feat3_att = feat3 + self.ca3(feat3)  # Residual connection
            else:
                feat3_att = feat3 + self.cbam3(feat3)  # Residual connection
            
            # Stage 4: Continue with CBAM attention (residual connection)
            feat4 = self.stage4_f(feat3_att)
            if self.use_ca:
                feat4_att = feat4 + self.ca4(feat4)  # Residual connection
            else:
                feat4_att = feat4 + self.cbam4(feat4)  # Residual connection
            
            # Store feature maps for potential use
            feature_maps = [feat3_att, feat4_att]
            
            # Use attended features for both tasks
            feat_flat = feat4_att.view(feat4_att.shape[0], -1)
            
            # Emotion branch: use output_layer -> embedding
            f_out = self.output_layer(feat4_att)      # (B, 2048)
            embedding = self.embedding(f_out)
            
            # Landmark branch (shares the same spatial features)
            pred_landmarks = self.landmark_layer(feat_flat)
        
        # Common output layers
        mean = self.mean_layer(embedding)
        logvar = self.logvar_layer(embedding)
        z0 = self.sample_z(mean, logvar)  # Noise vector for GAN
        z1 = self.c_layer(embedding)      # Emotion logits
        
        return mean, logvar, z0, z1, pred_landmarks


# ============================================================================
# Original VGG-style Encoder (Legacy, kept for compatibility)
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, img_size=224, fc_layer=512, latent_dim=10, noise_dim=100, e_ratio=0.2, scam_kernel=7):
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
        self.ccam1 = CCAM(64, e_ratio=e_ratio)
        self.scam1 = SCAM(64, kernel_size=scam_kernel)

        # =====================================================================
        # Stage 2 (/2 -> /4)
        # =====================================================================
        self.conv_block2_f = self._make_block(64, 128)
        self.conv_block2_l = self._make_block(64, 128)
        self.ccam2 = CCAM(128, e_ratio=e_ratio)
        self.scam2 = SCAM(128, kernel_size=scam_kernel)

        # =====================================================================
        # Stage 3 (/4 -> /8)
        # =====================================================================
        self.conv_block3_f = self._make_block(128, 256, layers=3)
        self.conv_block3_l = self._make_block(128, 256, layers=3)
        self.ccam3 = CCAM(256, e_ratio=e_ratio)
        self.scam3 = SCAM(256, kernel_size=scam_kernel)

        # =====================================================================
        # Stage 4 (/8 -> /16)
        # =====================================================================
        self.conv_block4_f = self._make_block(256, 512, layers=3)
        self.conv_block4_l = self._make_block(256, 512, layers=3)
        self.ccam4 = CCAM(512, e_ratio=e_ratio)
        self.scam4 = SCAM(512, kernel_size=scam_kernel)

        # =====================================================================
        # Stage 5 (/16 -> /32)
        # =====================================================================
        self.conv_block5_f = self._make_block(512, 512, layers=3)
        self.conv_block5_l = self._make_block(512, 512, layers=3)
        self.ccam5 = CCAM(512, e_ratio=e_ratio)
        self.scam5 = SCAM(512, kernel_size=scam_kernel)

        # =====================================================================
        # Face Head (Emotion & Latent Codes)
        # =====================================================================
        self.embedding = nn.Sequential(
            nn.Linear(512 * self.final_size * self.final_size, self.fc_layer),
            nn.BatchNorm1d(fc_layer),
            nn.Dropout(p=0.4),
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


class EmbeddingWrapper(nn.Module):
    """
    兼容性 Embedding：
    - 接受 (B, C, H, W) -> 自适应池化 -> (B, C) -> 投影到 target_dim -> fc_out
    - 或接受 (B, N) -> 若 N != target_dim 则动态创建 proj 将 N -> target_dim
    """
    def __init__(self, target_dim=2048, fc_out=512, dropout=0.4):
        super(EmbeddingWrapper, self).__init__()
        self.target_dim = target_dim
        self.fc_out_dim = fc_out
        self.dropout_p = dropout

        # 动态投影层（首次 forward 时创建并注册）
        self.proj = None
        # 固定输出层（以 target_dim 为输入）
        self.fc_out = nn.Linear(self.target_dim, self.fc_out_dim)
        self.bn = nn.BatchNorm1d(self.fc_out_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        # 支持 (B, C, H, W)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)  # (B, C)
        # 若输入为 (B, N) 且 N != target_dim，则动态创建投影层
        if x.dim() == 2 and x.size(1) != self.target_dim:
            if self.proj is None:
                self.proj = nn.Linear(x.size(1), self.target_dim).to(x.device)
            x = self.proj(x)
        # 现在 x.size(1) == target_dim
        x = self.fc_out(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
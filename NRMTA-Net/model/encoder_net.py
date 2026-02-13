import os
import torch
import torch.nn as nn
from .attention import HeteroCoAttentionModule


# ============================================================================
# IR-50 (ResNet-50 with ArcFace Structure) Building Blocks
# ============================================================================
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
# MobileFaceNet Building Blocks
# ============================================================================

class Conv_block(nn.Module):
    """Standard convolution block: Conv2d -> BN -> PReLU"""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    """Linear convolution block (no activation): Conv2d -> BN"""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Depth_Wise(nn.Module):
    """
    Depthwise separable convolution block for MobileFaceNet.
    Matches cunjian/pytorch_face_landmark implementation keys.
    """
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet for Facial Landmark Detection (FLD).
    Re-implemented to match cunjian/pytorch_face_landmark structure exactly.
    Returns multi-scale features for Co-Attention.
    """
    def __init__(self, embedding_size=512): # default was 136 in provided code but likely 136 or 512 depending on model
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # Output layer implementation (renamed to match weights)
        class OutputLayer(nn.Module):
             def __init__(self, embedding_size):
                 super(OutputLayer, self).__init__()
                 self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
                 self.flatten = Flatten()
                 self.linear = nn.Linear(512, embedding_size, bias=False)
                 self.bn = nn.BatchNorm1d(embedding_size)
             def forward(self, x):
                 x = self.conv_6_dw(x)
                 x = self.flatten(x)
                 x = self.linear(x)
                 x = self.bn(x)
                 return x

        self.output_layer = OutputLayer(embedding_size)
        
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        feat0 = out  # 56x56

        out = self.conv_23(out)
        out = self.conv_3(out)
        feat1 = out  # 28x28

        out = self.conv_34(out)
        out = self.conv_4(out)
        feat2 = out  # 14x14

        out = self.conv_45(out)
        out = self.conv_5(out)
        feat3 = out  # 7x7

        out = self.conv_6_sep(out)
        embedding = self.output_layer(out)
            
        return [feat0, feat1, feat2, feat3], embedding


# ============================================================================
# Encoder_Net: Heterogeneous Dual-Stream Encoder (IR50 + MobileFaceNet)
# ============================================================================

class Encoder_Net(nn.Module):
    """
    Heterogeneous Dual-Stream Encoder.

    - FER stream  : IR-50 backbone (ArcFace pretrained) for expression recognition.
    - FLD stream  : MobileFaceNet backbone for facial landmark detection.
    - Interaction : HeteroCoAttentionModule at each stage (bidirectional, CCAM+SCAM).

    IR-50 Stage layout [3, 4, 14, 3] with channels [64, 128, 256, 512].
    MobileFaceNet multi-scale features with channels [64, 64, 128, 128].

    Forward returns:
        fer_embedding : (B, 512) — for emotion classification / GAN.
        fld_output    : (B, 136) — 68 landmarks x 2.
    """
    def __init__(self, img_size=112, fer_embedding_dim=512, fld_embedding_size=136,
                 use_se=False, e_ratio=0.2, scam_kernel=7, align_dim=None):
        super(Encoder_Net, self).__init__()
        self.img_size = img_size
        self.fer_embedding_dim = fer_embedding_dim

        # Choose IR block type
        block = BottleneckIRSE if use_se else BottleneckIR
        num_blocks = [3, 4, 14, 3]  # IR-50

        # =================================================================
        # FER Backbone (IR-50) — split into stages for interleaved attention
        # =================================================================
        # Input layer (Stem)
        self.fer_input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        # Stage 1: 64ch, 112->56  (3 blocks)
        stage1 = []
        for i in range(num_blocks[0]):
            stage1.append(block(64, 64, 2 if i == 0 else 1))
        self.fer_layer1 = nn.Sequential(*stage1)

        # Stage 2: 64->128, 56->28  (4 blocks)
        stage2 = [block(64, 128, 2)]
        for _ in range(1, num_blocks[1]):
            stage2.append(block(128, 128, 1))
        self.fer_layer2 = nn.Sequential(*stage2)

        # Stage 3: 128->256, 28->14  (14 blocks)
        stage3 = [block(128, 256, 2)]
        for _ in range(1, num_blocks[2]):
            stage3.append(block(256, 256, 1))
        self.fer_layer3 = nn.Sequential(*stage3)

        # Stage 4: 256->512, 14->7  (3 blocks)
        stage4 = [block(256, 512, 2)]
        for _ in range(1, num_blocks[3]):
            stage4.append(block(512, 512, 1))
        self.fer_layer4 = nn.Sequential(*stage4)

        # =================================================================
        # FLD Backbone (MobileFaceNet)
        # =================================================================
        self.fld_backbone = MobileFaceNet(embedding_size=fld_embedding_size)

        # =================================================================
        # Co-Attention Modules (FLD guides FER at each stage)
        #   IR50 channels:          [64,  128, 256, 512]
        #   MobileFaceNet channels: [64,   64, 128, 128]
        # =================================================================
        if img_size % 16 != 0:
            raise ValueError(f"Unsupported img_size: {img_size}. Expected divisible by 16.")
        s1 = img_size // 2
        s2 = img_size // 4
        s3 = img_size // 8
        s4 = img_size // 16

        self.attn1 = HeteroCoAttentionModule(fer_channels=64,  fld_channels=64,  e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s1)
        self.attn2 = HeteroCoAttentionModule(fer_channels=128, fld_channels=64,  e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s2)
        self.attn3 = HeteroCoAttentionModule(fer_channels=256, fld_channels=128, e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s3)
        self.attn4 = HeteroCoAttentionModule(fer_channels=512, fld_channels=128, e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s4)

        # [Method 4] Attention Map Alignment
        # We assume attention masks are returned by HeteroCoAttentionModule directly.
        # No extra projection heads needed.
        self.fer_align = None
        self.fld_align = None

        # =================================================================
        # FER Output Head: BN -> Dropout -> Flatten -> FC -> BN  (=> 512-d)
        # =================================================================
        if img_size == 112:
            final_spatial = 7  # 112 / 2^4 = 7
        elif img_size == 224:
            final_spatial = 14
        else:
            raise ValueError(f"Unsupported img_size: {img_size}. Use 112 or 224.")

        # [ARCH CHANGE] Feature Fusion Bottleneck
        # Fuses FER (512) and FLD (512) features before classification
        # to force gradient flow into the landmark backbone.
        print("[Encoder_Net] Enabling Feature Fusion (FER+FLD -> Bottleneck -> Output)")
        self.fusion_bottleneck = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.fer_output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * final_spatial * final_spatial, fer_embedding_dim),
            nn.BatchNorm1d(fer_embedding_dim),
        )

    # ------------------------------------------------------------------
    # Dual Pretrained Weight Loading (Fixed Mapping)
    # ------------------------------------------------------------------
    def load_pretrained_weights(self, fer_path, fld_path):
        print("-" * 50)
        print(f"[Encoder_Net] Initializing weights...")

        # ========================= FER (IR-50) =========================
        if fer_path and os.path.isfile(fer_path):
            print(f"[Encoder_Net] Loading FER weights from: {fer_path}")
            state_dict = torch.load(fer_path, map_location='cpu')
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            
            clean_sd = {}
            for k, v in state_dict.items():
                nk = k
                if nk.startswith('module.'): nk = nk[7:]
                if nk.startswith('backbone.'): nk = nk[9:]
                clean_sd[nk] = v

            mapped_sd = {}
            for k, v in clean_sd.items():
                if k.startswith('input_layer.'):
                    mapped_sd[f"fer_input_layer.{k[len('input_layer.'):]}"] = v
                elif k.startswith('body.'):
                    parts = k.split('.')
                    try:
                        block_idx = int(parts[1])
                        rest = '.'.join(parts[2:])
                        if block_idx <= 2:
                            mapped_sd[f"fer_layer1.{block_idx}.{rest}"] = v
                        elif block_idx <= 6:
                            mapped_sd[f"fer_layer2.{block_idx - 3}.{rest}"] = v
                        elif block_idx <= 20:
                            mapped_sd[f"fer_layer3.{block_idx - 7}.{rest}"] = v
                        elif block_idx <= 23:
                            mapped_sd[f"fer_layer4.{block_idx - 21}.{rest}"] = v
                    except: continue
            
            res = self.load_state_dict(mapped_sd, strict=False)
            
            # --- MODIFIED: Print missing non-backbone keys as requested ---
            params_fer_backbone = [k for k in res.missing_keys if k.startswith('fer_layer') or k.startswith('fer_input')]
            params_others = [k for k in res.missing_keys if k not in params_fer_backbone]
            
            if len(params_fer_backbone) == 0:
                print(f"[Success] FER (IR-50) backbone loaded fully!")
            else:
                print(f"[Warning] FER Backbone partial missing: {len(params_fer_backbone)}")
            
            print(f"[Info] New layers initialized from scratch (Attentions/Heads): {len(params_others)}")
            if len(params_others) > 0:
               print(f"       Sample: {params_others[:3]} ...")

        # ======================== FLD (MobileFaceNet) ========================
        if fld_path and os.path.isfile(fld_path):
            print(f"[Encoder_Net] Loading FLD weights from: {fld_path}")
            fld_sd = torch.load(fld_path, map_location='cpu')
            if 'model_state_dict' in fld_sd: fld_sd = fld_sd['model_state_dict']
            if 'state_dict' in fld_sd: fld_sd = fld_sd['state_dict']
            
            # 1. Clean file keys
            clean_fld = {}
            for k, v in fld_sd.items():
                nk = k
                if nk.startswith('module.'): nk = nk[7:]
                if nk.startswith('fld_backbone.'): nk = nk[13:]
                clean_fld[nk] = fld_sd[k]
            
            # 2. Get model keys
            model_keys = list(self.fld_backbone.state_dict().keys())
            
            # 3. Strategy: Direct Load with new matching structure
            res = self.fld_backbone.load_state_dict(clean_fld, strict=False)
            
            missing = [k for k in res.missing_keys if "num_batches_tracked" not in k]
            # Ignore missing 'num_batches_tracked' which is common in older pth files

            if len(missing) > 0:
                print(f"[Warning] FLD partial missing keys: {len(missing)}")
                print(f"          First missing: {missing[:5]}")
                # Optional: Check if we have prefix issues? 
                # e.g. if Missing keys start with 'module.' or something we missed.
            else:
                 print(f"[Success] FLD (MobileFaceNet) weights loaded fully!")
        else:
            print(f"[Encoder_Net] WARNING: FLD weights not found at '{fld_path}'.")
        print("-" * 50)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, return_feats=False):
        """
        Interleaved dual-stream forward (bidirectional).

        Args:
            x: (B, 3, 112, 112)
        Returns:
            fer_embedding: (B, 512)  — expression feature vector.
            fld_output:    (B, 136)  — landmark coordinates.
        """
        # ---------- FER stem -------------------------------------------------
        fer_x = self.fer_input_layer(x)        # (B, 64, 112, 112)

        # ---------- FLD stem -------------------------------------------------
        fld_x = self.fld_backbone.conv1(x)     # (B, 64, 56, 56)
        fld_x = self.fld_backbone.conv2_dw(fld_x)

        align_feats = []

        # Stage 1
        fer_x = self.fer_layer1(fer_x)         # (B, 64, 56, 56)
        fer_x, fld_x, m_fer1, m_fld1 = self.attn1(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer1, m_fld1))

        # Stage 2
        fld_x = self.fld_backbone.conv_23(fld_x)
        fld_x = self.fld_backbone.conv_3(fld_x)  # (B, 64, 28, 28)
        fer_x = self.fer_layer2(fer_x)           # (B, 128, 28, 28)
        fer_x, fld_x, m_fer2, m_fld2 = self.attn2(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer2, m_fld2))

        # Stage 3
        fld_x = self.fld_backbone.conv_34(fld_x)
        fld_x = self.fld_backbone.conv_4(fld_x)  # (B, 128, 14, 14)
        fer_x = self.fer_layer3(fer_x)           # (B, 256, 14, 14)
        fer_x, fld_x, m_fer3, m_fld3 = self.attn3(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer3, m_fld3))

        # Stage 4
        fld_x = self.fld_backbone.conv_45(fld_x)
        fld_x = self.fld_backbone.conv_5(fld_x)  # (B, 128, 7, 7)
        fer_x = self.fer_layer4(fer_x)           # (B, 512, 7, 7)
        fer_x, fld_x, m_fer4, m_fld4 = self.attn4(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer4, m_fld4))

        # FLD embedding head
        fld_x = self.fld_backbone.conv_6_sep(fld_x)
        fld_output = self.fld_backbone.output_layer(fld_x)

        # FER embedding head (With Feature Fusion)
        # Gradient Flow: Loss -> fer_embedding -> fer_output_layer -> fusion_bottleneck -> [fer_x, fld_x]
        # This guarantees fld_x receives gradients from the emotion classification task.
        combined = torch.cat([fer_x, fld_x], dim=1)
        fused_x = self.fusion_bottleneck(combined)
        fer_embedding = self.fer_output_layer(fused_x)  # (B, 512)

        if return_feats:
            return fer_embedding, fld_output, align_feats
        return fer_embedding, fld_output


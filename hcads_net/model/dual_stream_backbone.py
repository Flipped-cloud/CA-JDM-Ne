import os
import torch
import torch.nn as nn
from .attention import HeteroCoAttentionModule


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
    def __init__(self, embedding_size=512):
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
# DualStreamBackbone: Heterogeneous Dual-Stream Backbone (IR50 + MobileFaceNet)
# ============================================================================

class DualStreamBackbone(nn.Module):
    """
    Heterogeneous Dual-Stream Backbone.

    - FER stream  : IR-50 backbone (ArcFace pretrained) for expression recognition.
    - FLD stream  : MobileFaceNet backbone for facial landmark detection.
    - Interaction : HeteroCoAttentionModule at each stage (bidirectional, CCAM+SCAM).

    IR-50 Stage layout [3, 4, 14, 3] with channels [64, 128, 256, 512].
    MobileFaceNet multi-scale features with channels [64, 64, 128, 128].

    Forward returns:
        fer_embedding : (B, 512) — for emotion classification.
        fld_output    : (B, 136) — 68 landmarks x 2.
    """
    def __init__(
        self,
        img_size: int = 112,
        fer_embedding_dim: int = 512,
        fld_embedding_size: int = 136,
        use_se: bool = False,
        e_ratio: float = 0.2,
        scam_kernel: int = 7,
    ):
        super().__init__()
        self.img_size = int(img_size)
        self.fer_embedding_dim = int(fer_embedding_dim)

        if self.img_size not in (112, 224):
            raise ValueError(f"Unsupported img_size: {self.img_size}. Use 112 or 224.")

        block = BottleneckIRSE if use_se else BottleneckIR
        num_blocks = [3, 4, 14, 3]  # IR-50

        self.fer_input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        stage1 = []
        for i in range(num_blocks[0]):
            stage1.append(block(64, 64, 2 if i == 0 else 1))
        self.fer_layer1 = nn.Sequential(*stage1)

        stage2 = [block(64, 128, 2)]
        for _ in range(1, num_blocks[1]):
            stage2.append(block(128, 128, 1))
        self.fer_layer2 = nn.Sequential(*stage2)

        stage3 = [block(128, 256, 2)]
        for _ in range(1, num_blocks[2]):
            stage3.append(block(256, 256, 1))
        self.fer_layer3 = nn.Sequential(*stage3)

        stage4 = [block(256, 512, 2)]
        for _ in range(1, num_blocks[3]):
            stage4.append(block(512, 512, 1))
        self.fer_layer4 = nn.Sequential(*stage4)

        self.fld_backbone = MobileFaceNet(embedding_size=int(fld_embedding_size))

        s1 = self.img_size // 2
        s2 = self.img_size // 4
        s3 = self.img_size // 8
        s4 = self.img_size // 16

        self.attn1 = HeteroCoAttentionModule(fer_channels=64, fld_channels=64, e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s1)
        self.attn2 = HeteroCoAttentionModule(fer_channels=128, fld_channels=64, e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s2)
        self.attn3 = HeteroCoAttentionModule(fer_channels=256, fld_channels=128, e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s3)
        self.attn4 = HeteroCoAttentionModule(fer_channels=512, fld_channels=128, e_ratio=e_ratio, scam_kernel=scam_kernel, spatial_size=s4)

        final_spatial = 7 if self.img_size == 112 else 14
        self.fer_output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * final_spatial * final_spatial, self.fer_embedding_dim),
            nn.BatchNorm1d(self.fer_embedding_dim),
        )

    def load_pretrained_weights(self, fer_path: str = "", fld_path: str = ""):
        """Optionally load pretrained weights for IR50 (FER) and MobileFaceNet (FLD)."""
        if fer_path and os.path.isfile(fer_path):
            print(f"[DualStreamBackbone] Loading FER weights from: {fer_path}")
            state_dict = torch.load(fer_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            clean_sd = {}
            for k, v in state_dict.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[7:]
                if nk.startswith("backbone."):
                    nk = nk[9:]
                clean_sd[nk] = v

            mapped_sd = {}
            for k, v in clean_sd.items():
                if k.startswith("input_layer."):
                    mapped_sd[f"fer_input_layer.{k[len('input_layer.'):]}" ] = v
                elif k.startswith("body."):
                    parts = k.split(".")
                    try:
                        block_idx = int(parts[1])
                    except Exception:
                        continue
                    rest = ".".join(parts[2:])
                    if block_idx <= 2:
                        mapped_sd[f"fer_layer1.{block_idx}.{rest}"] = v
                    elif block_idx <= 6:
                        mapped_sd[f"fer_layer2.{block_idx - 3}.{rest}"] = v
                    elif block_idx <= 20:
                        mapped_sd[f"fer_layer3.{block_idx - 7}.{rest}"] = v
                    elif block_idx <= 23:
                        mapped_sd[f"fer_layer4.{block_idx - 21}.{rest}"] = v

            res = self.load_state_dict(mapped_sd, strict=False)
            missing_backbone = [k for k in res.missing_keys if k.startswith("fer_layer") or k.startswith("fer_input_layer")]
            if missing_backbone:
                print(f"[DualStreamBackbone] WARNING: FER backbone missing keys: {len(missing_backbone)}")
            else:
                print("[DualStreamBackbone] FER backbone loaded.")
        elif fer_path:
            print(f"[DualStreamBackbone] WARNING: FER weights not found at '{fer_path}'.")

        if fld_path and os.path.isfile(fld_path):
            print(f"[DualStreamBackbone] Loading FLD weights from: {fld_path}")
            fld_sd = torch.load(fld_path, map_location="cpu")
            if isinstance(fld_sd, dict) and "model_state_dict" in fld_sd:
                fld_sd = fld_sd["model_state_dict"]
            if isinstance(fld_sd, dict) and "state_dict" in fld_sd:
                fld_sd = fld_sd["state_dict"]

            clean_fld = {}
            for k, v in fld_sd.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[7:]
                if nk.startswith("fld_backbone."):
                    nk = nk[13:]
                clean_fld[nk] = v

            res = self.fld_backbone.load_state_dict(clean_fld, strict=False)
            missing = [k for k in res.missing_keys if "num_batches_tracked" not in k]
            if missing:
                print(f"[DualStreamBackbone] WARNING: FLD missing keys: {len(missing)}")
            else:
                print("[DualStreamBackbone] FLD backbone loaded.")
        elif fld_path:
            print(f"[DualStreamBackbone] WARNING: FLD weights not found at '{fld_path}'.")

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        """Forward pass.

        Args:
            x: (B, 3, img_size, img_size)
            return_feats: whether to return (mask_fer, mask_fld) pairs for alignment loss
        """
        fer_x = self.fer_input_layer(x)

        fld_x = self.fld_backbone.conv1(x)
        fld_x = self.fld_backbone.conv2_dw(fld_x)

        align_feats = []

        fer_x = self.fer_layer1(fer_x)
        fer_x, fld_x, m_fer1, m_fld1 = self.attn1(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer1, m_fld1))

        fld_x = self.fld_backbone.conv_23(fld_x)
        fld_x = self.fld_backbone.conv_3(fld_x)
        fer_x = self.fer_layer2(fer_x)
        fer_x, fld_x, m_fer2, m_fld2 = self.attn2(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer2, m_fld2))

        fld_x = self.fld_backbone.conv_34(fld_x)
        fld_x = self.fld_backbone.conv_4(fld_x)
        fer_x = self.fer_layer3(fer_x)
        fer_x, fld_x, m_fer3, m_fld3 = self.attn3(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer3, m_fld3))

        fld_x = self.fld_backbone.conv_45(fld_x)
        fld_x = self.fld_backbone.conv_5(fld_x)
        fer_x = self.fer_layer4(fer_x)
        fer_x, fld_x, m_fer4, m_fld4 = self.attn4(fer_x, fld_x)
        if return_feats:
            align_feats.append((m_fer4, m_fld4))

        fld_x = self.fld_backbone.conv_6_sep(fld_x)
        fld_output = self.fld_backbone.output_layer(fld_x)

        fer_embedding = self.fer_output_layer(fer_x)

        if return_feats:
            return fer_embedding, fld_output, align_feats
        return fer_embedding, fld_output

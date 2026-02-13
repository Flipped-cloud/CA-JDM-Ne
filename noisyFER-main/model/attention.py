import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CMCNN-style Attention Modules (from CBAM)
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (from CBAM/CMCNN)
    Reference: https://github.com/luuuyi/CBAM.PyTorch
    
    使用全局平均池化和最大池化来生成通道注意力权重
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)  # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (from CBAM/CMCNN)
    Reference: https://github.com/luuuyi/CBAM.PyTorch
    
    使用通道维度的平均池化和最大池化来生成空间注意力权重
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn(x_cat)
        return self.sigmoid(x_cat)  # (B, 1, H, W)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    结合通道注意力和空间注意力的完整模块
    
    使用方式: 
    x_att = x + CBAM(x)  # 残差连接
    或
    x_att = x * CBAM(x)  # 直接加权
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Channel attention
        x = x * self.ca(x)
        # Spatial attention
        x = x * self.sa(x)
        return x


# ============================================================================
# Coordinate Attention (CA)
# Reference: Coordinate Attention for Efficient Mobile Network Design
# ============================================================================

class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module
    Generates direction-aware attention maps along H and W.
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = HSwish()

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # Coordinate information embedding
        x_h = x.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, 1, W)

        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


# ============================================================================
# Co-Attention Modules (for multi-stream architectures)
# ============================================================================

class CCAM(nn.Module):
    """
    Channel Correlation Attention Module (CCAM)

    目标（对齐 CMCNN 论文 CCAM + 结合方案）：
    - 输入两路特征 x_face, x_land (B,C,H,W)
    - 先对每个通道做空间嵌入（HW -> D），得到 Fe/Fl
    - 计算相关性矩阵 S = Fe * Fl^T / sqrt(D)
    - 按论文分别沿不同维度 softmax 并求和得到 A_e / A_l
    - 输出两路加权后的特征 (x_face', x_land')
    """
    def __init__(self, channel: int, e_ratio: float = 0.2, spatial_hw=None):
        super().__init__()
        self.e_ratio = e_ratio
        # 按空间尺寸缓存 MLP，避免每次 forward 重新构建
        self.face_mlps = nn.ModuleDict()
        self.land_mlps = nn.ModuleDict()

        if spatial_hw is not None:
            for hw in spatial_hw:
                key = str(int(hw))
                if key not in self.face_mlps:
                    self.face_mlps[key] = self._build_mlp(int(hw))
                if key not in self.land_mlps:
                    self.land_mlps[key] = self._build_mlp(int(hw))

    def _build_mlp(self, in_feature: int):
        out_feature = max(1, int(in_feature * self.e_ratio))
        return nn.Sequential(
            nn.Linear(in_feature, out_feature, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(out_feature, out_feature, bias=True),
        )

    def forward(self, x_face: torch.Tensor, x_land: torch.Tensor):
        # x_face/x_land: (B,C,H,W)
        b, c, h, w = x_face.shape
        hw = h * w
        key = str(hw)

        if key not in self.face_mlps:
            self.face_mlps[key] = self._build_mlp(hw).to(x_face.device)
        if key not in self.land_mlps:
            self.land_mlps[key] = self._build_mlp(hw).to(x_land.device)

        # (B,C,H,W) -> (B*C, HW) -> (B,C,D)
        xe2 = x_face.view(b * c, -1)
        xe2 = self.face_mlps[key](xe2).view(b, c, -1)

        xl2 = x_land.view(b * c, -1)
        xl2 = self.land_mlps[key](xl2).view(b, c, -1)

        # 相关性矩阵 S
        cc = torch.bmm(xe2, xl2.transpose(2, 1)) / math.sqrt(xe2.size(-1))

        # A_e / A_l（按论文在不同维度 softmax）
        a_e = torch.softmax(cc, dim=1).sum(dim=2).unsqueeze(-1).unsqueeze(-1)
        a_l = torch.softmax(cc, dim=2).sum(dim=1).unsqueeze(-1).unsqueeze(-1)

        x_face_att = x_face * a_e
        x_land_att = x_land * a_l
        return x_face_att, x_land_att


class SCAM(nn.Module):
    """
    Spatial Co-Attention Module (SCAM)

    目标（对齐 CMCNN 论文 SCAM + 结合方案）：
    - 输入两路特征 x_face, x_land (B,C,H,W)
    - 对拼接特征做 max/avg pooling 得到 F_max/F_avg
    - 拼接后经 7x7 卷积 + sigmoid 得到共享空间掩码 A_s
    - 用同一 A_s 同时加权两路特征
    """
    def __init__(self, channel: int, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_face: torch.Tensor, x_land: torch.Tensor):
        # 拼接： (B,2C,H,W)
        x = torch.cat([x_face, x_land], dim=1)

        # max/avg pooling along channel
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)

        a_s = self.sigmoid(self.conv(x_att))

        x_face_att = x_face * a_s
        x_land_att = x_land * a_s
        return x_face_att, x_land_att


# ============================================================================
# Co-Attention Module (for heterogeneous dual-stream: IR50 + MobileFaceNet)
# ============================================================================

class CoAttentionModule(nn.Module):
    """
    Co-Attention Module for heterogeneous dual-stream architecture.

    利用 FLD (人脸关键点) 分支的特征生成空间注意力 Mask，
    作用于 FER (表情识别) 分支的特征上，以残差连接方式输出。

    由于 IR50 与 MobileFaceNet 是异构网络，二者在相同深度的
    通道数和空间尺寸可能不同，因此模块内部会：
    1. 检查空间尺寸是否匹配，不匹配时对 FLD 特征做 interpolate。
    2. 使用 Conv->BN->ReLU->Conv->Sigmoid 生成空间注意力掩码。

    Args:
        fld_channels (int): FLD 分支特征的通道数。
        fer_channels (int): FER 分支特征的通道数（本模块未直接使用，
                            但保留以便后续扩展通道对齐等操作）。
    """
    def __init__(self, fld_channels: int, fer_channels: int):
        super(CoAttentionModule, self).__init__()
        self.fld_channels = fld_channels
        self.fer_channels = fer_channels

        # 使用两层卷积 + BN + ReLU 生成更强的空间注意力掩码
        mid_channels = max(1, fld_channels // 4)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(fld_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_fer: torch.Tensor, f_fld: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_fer: FER 分支特征, shape (B, fer_channels, H_fer, W_fer)
            f_fld: FLD 分支特征, shape (B, fld_channels, H_fld, W_fld)

        Returns:
            增强后的 FER 特征, shape 与 f_fer 一致 (B, fer_channels, H_fer, W_fer)
        """
        # 空间尺寸对齐：将 FLD 特征缩放到与 FER 特征相同的 H, W
        if f_fld.shape[2:] != f_fer.shape[2:]:
            f_fld = F.interpolate(
                f_fld,
                size=f_fer.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        # 生成空间注意力掩码 (B, 1, H, W)
        mask = self.spatial_attn(f_fld)

        # 残差连接: output = f_fer * mask + f_fer
        output = f_fer * mask + f_fer
        return output


class HeteroCoAttentionModule(nn.Module):
    """
    Heterogeneous co-attention block aligned with CMCNN-style CCAM + SCAM.

    This block aligns FLD features to FER channels (if needed), then applies
    CCAM (channel correlation) and SCAM (spatial co-attention). The two
    attended FER features are blended with learnable weights and a residual.
    """
    def __init__(self, fer_channels: int, fld_channels: int, e_ratio: float = 0.2, scam_kernel: int = 7, spatial_size: int = None):
        super().__init__()
        self.fer_channels = fer_channels
        self.fld_channels = fld_channels

        if fer_channels != fld_channels:
            self.fld_to_fer = nn.Sequential(
                nn.Conv2d(fld_channels, fer_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fer_channels),
                nn.ReLU(inplace=True),
            )
            self.fer_to_fld = nn.Sequential(
                nn.Conv2d(fer_channels, fld_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fld_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.fld_to_fer = nn.Identity()
            self.fer_to_fld = nn.Identity()

        spatial_hw = None
        if spatial_size is not None:
            spatial_hw = [int(spatial_size) * int(spatial_size)]
        
        self.ccam = CCAM(fer_channels, e_ratio=e_ratio, spatial_hw=spatial_hw)
        self.scam = SCAM(fer_channels, kernel_size=scam_kernel)

        # [Method 4] Independent Spatial Attention for Alignment
        self.sa_fer = SpatialAttention(kernel_size=scam_kernel)
        self.sa_fld = SpatialAttention(kernel_size=scam_kernel)

        # CCAM/SCAM 内部融合权重
        self.alpha_f = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        self.alpha_l = nn.Parameter(torch.FloatTensor([0.5, 0.5]))

        # CMCNN 风格 cross-stitch 2×2 融合矩阵
        self.cs = nn.Parameter(torch.tensor([[1.0, 0.0],
                                             [0.0, 1.0]], dtype=torch.float32))

    def forward(self, f_fer: torch.Tensor, f_fld: torch.Tensor):
        f_fld_aligned = self.fld_to_fer(f_fld)

        # [Method 4] Compute separate spatial attention maps for alignment
        # Use Activation Energy (mean squared) as a robust proxy for spatial attention
        # since we want to align "where natural attention is" without extra learnable parameters.
        # (B, C, H, W) -> (B, 1, H, W)
        mask_fer = torch.mean(f_fer.pow(2), dim=1, keepdim=True)
        mask_fld = torch.mean(f_fld_aligned.pow(2), dim=1, keepdim=True)
        
        # L2 Normalize the spatial maps to make alignment scale-invariant
        # This ensures we align the "shape" of attention, not the magnitude.
        B, _, H, W = mask_fer.shape
        mask_fer = F.normalize(mask_fer.view(B, -1), dim=1).view(B, 1, H, W)
        mask_fld = F.normalize(mask_fld.view(B, -1), dim=1).view(B, 1, H, W)

        f_ccam, l_ccam = self.ccam(f_fer, f_fld_aligned)
        f_scam, l_scam = self.scam(f_fer, f_fld_aligned)

        w_f = torch.softmax(self.alpha_f, dim=0)
        w_l = torch.softmax(self.alpha_l, dim=0)

        f_att = w_f[0] * f_ccam + w_f[1] * f_scam
        l_att = w_l[0] * l_ccam + w_l[1] * l_scam

        # cross-stitch 融合（双向）
        f_mix = self.cs[0, 0] * f_att + self.cs[0, 1] * l_att
        l_mix = self.cs[1, 0] * f_att + self.cs[1, 1] * l_att

        # 残差 + 反向对齐
        out_fer = f_mix + f_fer
        out_fld_aligned = l_mix + f_fld_aligned
        out_fld = self.fer_to_fld(out_fld_aligned)

        return out_fer, out_fld, mask_fer, mask_fld
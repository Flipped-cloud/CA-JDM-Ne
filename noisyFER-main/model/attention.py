import torch
import torch.nn as nn
import torch.nn.functional as F

class CCAM(nn.Module):
    """
    Channel Correlation Attention Module (CCAM)
    计算两个特征流（Face流 和 Landmark流）之间的通道相关性
    """
    def __init__(self, channel, reduction=16):
        super(CCAM, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 共享权重的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_face, x_land):
        # 1. 压缩空间维度: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        b, c, _, _ = x_face.size()
        y_face = self.avg_pool(x_face).view(b, c)
        y_land = self.avg_pool(x_land).view(b, c)

        # 2. 通过 MLP 计算通道权重
        # 这里的关键是：Face流的权重可能受Landmark流影响，反之亦然
        # 这里采用简单的相加融合策略（参考 CMCNN 原理）
        y_fusion = self.mlp(y_face + y_land)
        
        # 3. 生成权重
        att = self.sigmoid(y_fusion).view(b, c, 1, 1)
        
        # 4. 加权
        return x_face * att, x_land * att

class SCAM(nn.Module):
    """
    Spatial Co-Attention Module (SCAM)
    计算空间上的注意力掩码
    """
    def __init__(self, channel):
        super(SCAM, self).__init__()
        # 使用 1x1 卷积融合两个流的特征
        # 输入通道是 2 * channel (因为是 concat)
        self.conv = nn.Sequential(
            nn.Conv2d(channel * 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_face, x_land):
        # 1. 在通道维度拼接: (B, 2C, H, W)
        x_cat = torch.cat([x_face, x_land], dim=1)
        
        # 2. 生成空间掩码: (B, 1, H, W)
        att = self.conv(x_cat)
        
        # 3. 加权
        return x_face * att, x_land * att
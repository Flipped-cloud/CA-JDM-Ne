import torch
import torch.nn as nn
import torch.nn.functional as F


class CCAM(nn.Module):
    """
    Channel Correlation Attention Module (CCAM)

    目标（对齐结合方案第4部分表述）：
    - 输入两路特征 x_face, x_land (B,C,H,W)
    - 计算两路之间的“通道相关性”并生成两套通道权重：
        W_ca: 用于 face 流
        W_cb: 用于 land 流
    - 输出两路加权后的特征 (x_face', x_land')，接口保持不变
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden = max(1, channel // reduction)

        # 共享的相关性编码：输入拼接后的 (2C) -> hidden
        self.fc_shared = nn.Sequential(
            nn.Linear(channel * 2, hidden, bias=False),
            nn.ReLU(inplace=True),
        )
        # 两个分支头：hidden -> C，分别输出 W_ca / W_cb
        self.fc_face = nn.Linear(hidden, channel, bias=False)
        self.fc_land = nn.Linear(hidden, channel, bias=False)

    def forward(self, x_face: torch.Tensor, x_land: torch.Tensor):
        # x_face/x_land: (B,C,H,W)
        b, c, _, _ = x_face.shape

        # GAP 得到通道描述子： (B,C)
        f = self.avg_pool(x_face).view(b, c)
        l = self.avg_pool(x_land).view(b, c)

        # 通道“相关性”编码：拼接形成 (B,2C)
        u = torch.cat([f, l], dim=1)

        h = self.fc_shared(u)  # (B,hidden)

        # 两套权重 (B,C) -> (B,C,1,1)
        w_ca = torch.sigmoid(self.fc_face(h)).view(b, c, 1, 1)
        w_cb = torch.sigmoid(self.fc_land(h)).view(b, c, 1, 1)

        # 分别加权两流（保持接口不变）
        x_face_att = x_face * w_ca
        x_land_att = x_land * w_cb
        return x_face_att, x_land_att


class SCAM(nn.Module):
    """
    Spatial Co-Attention Module (SCAM)

    目标：
    - 输入两路特征 x_face, x_land (B,C,H,W)
    - 基于两路拼接特征生成两套空间注意力掩码：
        W_sa: face 空间掩码
        W_sb: land 空间掩码
    - 输出两路加权后的特征 (x_face', x_land')，接口保持不变
    """
    def __init__(self, channel: int):
        super().__init__()
        # 由 2C 通道融合后，输出 2 个空间掩码（face/land 各一个）
        self.conv = nn.Conv2d(channel * 2, 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x_face: torch.Tensor, x_land: torch.Tensor):
        # 拼接： (B,2C,H,W)
        x = torch.cat([x_face, x_land], dim=1)

        # (B,2,H,W) -> sigmoid
        m = torch.sigmoid(self.conv(x))

        # 拆成两张 mask： (B,1,H,W)
        w_sa = m[:, 0:1, :, :]
        w_sb = m[:, 1:2, :, :]

        x_face_att = x_face * w_sa
        x_land_att = x_land * w_sb
        return x_face_att, x_land_att
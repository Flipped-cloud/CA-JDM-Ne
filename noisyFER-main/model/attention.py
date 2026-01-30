import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, channel: int, e_ratio: float = 0.2):
        super().__init__()
        self.e_ratio = e_ratio
        # 按空间尺寸缓存 MLP，避免每次 forward 重新构建
        self.face_mlps = nn.ModuleDict()
        self.land_mlps = nn.ModuleDict()

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
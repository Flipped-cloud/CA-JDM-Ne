import torch
import torch.nn as nn
import math

class WingLoss(nn.Module):
    """
    Wing Loss for robust facial landmark localisation.
    参考文献: 'Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks'
    
    公式:
    loss = w * ln(1 + |x|/epsilon)   if |x| < w
    loss = |x| - C                   otherwise
    """
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        # C = w - w * ln(1 + w/epsilon)
        self.C = w * (1.0 - math.log(1.0 + w / epsilon))

    def forward(self, predictions, targets):
        # predictions: (B, 136) or (B, 68, 2)
        # targets: (B, 136) or (B, 68, 2)
        
        # 计算差值绝对值
        x = predictions - targets
        abs_x = torch.abs(x)
        
        # 判断哪些点在 wing 范围内 (|x| < w)
        mask = abs_x < self.w
        
        # 计算 Loss
        # Part 1: log 部分 (针对小误差，梯度更大)
        loss_wing = self.w * torch.log(1.0 + abs_x / self.epsilon)
        
        # Part 2: linear 部分 (针对大误差，梯度恒定)
        loss_linear = abs_x - self.C
        
        # 组合
        loss = torch.where(mask, loss_wing, loss_linear)
        
        return torch.mean(loss)
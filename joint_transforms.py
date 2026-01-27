import random
import torchvision.transforms.functional as F
from PIL import Image
import torch
import numpy as np

"""
Day 1 核心模块 (V2.0)：联合变换 (Joint Transforms) - 适配 68 点关键点
功能：
1. 图像与关键点同步 Resize / Crop。
2. 语义级翻转：当图片翻转时，自动交换左右对称的关键点索引（如左眼变右眼）。
"""

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, landmarks):
        for t in self.transforms:
            image, landmarks = t(image, landmarks)
        return image, landmarks

class Resize(object):
    def __init__(self, size):
        # size: (h, w) tuple or int
        self.size = size

    def __call__(self, image, landmarks):
        w_old, h_old = image.size
        
        # 执行图片 Resize
        image = F.resize(image, self.size)
        w_new, h_new = image.size
        
        # 执行关键点坐标缩放
        # landmarks shape: [136] (x0, y0, x1, y1, ... x67, y67)
        landmarks = landmarks.clone()
        landmarks[0::2] = landmarks[0::2] * (w_new / w_old) # x scale
        landmarks[1::2] = landmarks[1::2] * (h_new / h_old) # y scale
        
        return image, landmarks

class RandomHorizontalFlip(object):
    """
    针对 68 点关键点的智能翻转。
    需要传入 mirror_indices 列表，用于交换左右对应的点。
    """
    def __init__(self, p=0.5):
        self.p = p
        # 68 点关键点的左右对称索引映射 (Dlib/Multi-PIE 标准格式)
        # 左眉(22-26) <-> 右眉(17-21)
        # 左眼(42-47) <-> 右眼(36-41)
        # 等等...
        self.mirror_indices = [
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, # Jaw (0-16)
            26, 25, 24, 23, 22, # Right Brow (17-21) <-> Left Brow
            21, 20, 19, 18, 17, # Left Brow (22-26) <-> Right Brow
            27, 28, 29, 30,     # Nose Bridge (27-30) - usually vertical, but some datasets might swap logic, usually self-mapping
            35, 34, 33, 32, 31, # Nose Nostrils (31-35)
            45, 44, 43, 42, 47, 46, # Right Eye (36-41) <-> Left Eye (Logic: 36<->45, etc. need precise mapping)
            39, 38, 37, 36, 41, 40, # Left Eye (42-47) <-> Right Eye
            54, 53, 52, 51, 50, 49, 48, # Outer Mouth (48-54) - lip corners swap
            59, 58, 57, 56, 55,         # Outer Mouth lower
            64, 63, 62, 61, 60,         # Inner Mouth
            67, 66, 65                  # Inner Mouth
        ]
        
        # 精确的 68 点左右互换 Map (基于 0-indexed)
        # 参考 standard 68-point mirroring
        self.flip_map = dict()
        
        # 构造映射对: (i, j) 表示索引 i 和 j 互换
        pairs = [
            (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), # Jaw
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22), # Brows
            (31, 35), (32, 34), # Nose
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46), # Eyes (Standard 68 pt order needs care)
            (48, 54), (49, 53), (50, 52), (59, 55), (58, 56), # Outer Mouth
            (60, 64), (61, 63), (67, 65) # Inner Mouth
        ]
        # 填充 Map
        for i, j in pairs:
            self.flip_map[i] = j
            self.flip_map[j] = i
        # 剩下的点 (如 8, 27, 28, 29, 30, 33, 51, 57, 62, 66) 是中心点，不需要交换索引，只需要翻转坐标

    def __call__(self, image, landmarks):
        if random.random() < self.p:
            w, h = image.size
            # 1. 图像翻转
            image = F.hflip(image)
            
            # 2. 关键点坐标数学翻转
            landmarks = landmarks.clone()
            landmarks[0::2] = w - landmarks[0::2]
            
            # 3. 关键点语义索引交换 (Semantic Index Swapping)
            # 这一步对于 68 点至关重要！
            # 将 Tensor 转为 Reshape 后的列表方便操作 [68, 2]
            lmk_reshaped = landmarks.view(-1, 2)
            new_lmk = lmk_reshaped.clone()
            
            for i in range(68):
                if i in self.flip_map:
                    # 如果该点有对称点，取对称点的坐标
                    target_idx = self.flip_map[i]
                    new_lmk[i] = lmk_reshaped[target_idx]
                else:
                    # 中心点，保持原索引，坐标已翻转
                    pass
            
            return image, new_lmk.view(-1)
            
        return image, landmarks

class ToTensor(object):
    def __call__(self, image, landmarks):
        image = F.to_tensor(image)
        return image, landmarks

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, landmarks):
        image = F.normalize(image, self.mean, self.std)
        return image, landmarks
"""
RAF-DB 数据加载器 (IR50专用版)
适配模块4要求：
1. 支持 112x112 / 224x224 输入分辨率
2. 关键点归一化到 [0, 1]
3. 完整数据增强: RandomHorizontalFlip, RandomCrop, ColorJitter
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
import joint_transforms as JT


class RAFDataset_IR50(data.Dataset):
    """
    RAF-DB 数据集加载器 (IR50 专用)
    
    特性:
    - 支持 112x112 (标准) 或 224x224 (高分辨率) 输入
    - 图像与关键点同步变换
    - 关键点归一化到 [0, 1] 范围
    - 完整数据增强策略
    - 噪声标签注入 (训练集)
    
    参数:
    - args: 配置对象，需包含以下属性:
        * data_path: 图像根目录
        * img_size: 输入尺寸 (112 或 224)
        * num_classes: 类别数 (RAF-DB为7)
        * num_landmarks: 关键点数 (68)
        * noise_ratio: 噪声比例 (0.0-1.0)
        * raf_train_csv: 训练集CSV路径
        * raf_val_csv: 验证集CSV路径
    - phase: 'train' 或 'val'
    - transform: 自定义transform (可选)
    - use_augmentation: 是否使用数据增强 (默认True)
    """
    
    def __init__(self, args, phase, transform=None, use_augmentation=True):
        self.phase = phase
        self.args = args
        self.image_root = args.data_path
        self.use_augmentation = use_augmentation
        
        # -----------------------------------------------------------
        # 1. 定义联合变换 (图像 + 关键点同步)
        # -----------------------------------------------------------
        self.transform = transform
        if self.transform is None:
            if phase == "train" and use_augmentation:
                # 训练集：完整数据增强
                self.joint_transform = JT.Compose([
                    JT.Resize((args.img_size, args.img_size)),      # Resize to 112 or 224
                    JT.RandomHorizontalFlip(p=0.5),                # 翻转 (关键点自动对称交换)
                    JT.RandomCrop(args.img_size, padding=4),       # 随机裁剪 (padding增加多样性)
                    JT.ColorJitter(                                 # 颜色抖动
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1
                    ),
                    JT.ToTensor(),                                  # 转为 Tensor
                    JT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
                ])
            else:
                # 验证集：仅 Resize + ToTensor + Normalize
                self.joint_transform = JT.Compose([
                    JT.Resize((args.img_size, args.img_size)),
                    JT.ToTensor(),
                    JT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        else:
            # 用户自定义 transform
            self.joint_transform = None
        
        # -----------------------------------------------------------
        # 2. 加载数据
        # -----------------------------------------------------------
        # 2.1 确定 CSV 路径
        if phase == "train":
            csv_file = getattr(args, "raf_train_csv", None)
        else:
            csv_file = getattr(args, "raf_val_csv", None)
        
        # 向后兼容：如果没有显式指定，尝试从 image_root 查找
        if not csv_file:
            phase_csv = f"{phase}.csv"
            candidates = [
                os.path.join(self.image_root, phase_csv),
                os.path.join(os.path.dirname(self.image_root), phase_csv),
            ]
            csv_file = next((p for p in candidates if os.path.exists(p)), None)
        
        if not csv_file or not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"CSV not found: {csv_file}. "
                f"Set --raf_train_csv/--raf_val_csv explicitly. "
                f"image_root={self.image_root} phase={phase}"
            )
        
        print(f"[RAF-IR50] Loading {phase} data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # 2.2 解析图像路径
        image_col = None
        for cand in ("image_id", "image", "path", "file", "img", "img_path", "image_path"):
            if cand in df.columns:
                image_col = cand
                break
        if image_col is None:
            image_col = df.columns[0]  # 默认第一列
        
        self.file_paths = []
        for v in df[image_col].values:
            s = str(v).strip()
            if s.isdigit():
                s = f"{s}.jpg"
            elif not os.path.splitext(s)[1]:
                s = f"{s}.jpg"
            
            img_path = s if os.path.isabs(s) else os.path.join(self.image_root, s)
            self.file_paths.append(img_path)
        
        # 2.3 解析标签 (兼容 1-7 或 0-6)
        label_col = "label" if "label" in df.columns else None
        if label_col is None:
            if len(df.columns) >= 2:
                label_col = df.columns[1]
            else:
                raise KeyError(f"Label column not found in CSV: {csv_file}")
        
        raw_labels = df[label_col].values.astype(np.int64)
        # RAF-DB 原始标签为 1-7，转换为 0-6
        if raw_labels.max() == 7 and raw_labels.min() >= 1:
            clean_labels = raw_labels - 1
        else:
            clean_labels = raw_labels
        
        self.clean_labels = clean_labels
        
        # 2.4 注入噪声 (仅训练集)
        if phase == "train":
            self.train_labels = self.inject_noise(
                self.clean_labels, 
                noise_ratio=args.noise_ratio, 
                num_classes=args.num_classes
            )
        else:
            self.train_labels = self.clean_labels
        
        # 2.5 加载关键点数据
        # 检查 CSV 中是否有关键点列 (x_0, y_0, ..., x_67, y_67)
        x_cols = [f"x_{i}" for i in range(args.num_landmarks)]
        y_cols = [f"y_{i}" for i in range(args.num_landmarks)]
        
        if all(c in df.columns for c in (x_cols + y_cols)):
            # 读取关键点 (原始像素坐标)
            lmk = df[x_cols + y_cols].values.astype(np.float32)
            self.landmarks_data = lmk
            print(f"[RAF-IR50] Loaded {args.num_landmarks} landmarks from CSV")
        else:
            # 如果 CSV 中没有关键点，使用零向量占位
            self.landmarks_data = np.zeros(
                (len(self.file_paths), args.num_landmarks * 2), 
                dtype=np.float32
            )
            print(f"[RAF-IR50] Warning: No landmarks found in CSV, using zeros")
        
        print(f"[RAF-IR50] Loaded {len(self.file_paths)} images for {phase} set")
    
    def inject_noise(self, labels, noise_ratio, num_classes=7):
        """
        为训练集注入噪声标签 (模拟真实场景)
        """
        labels = np.asarray(labels, dtype=np.int64).copy()
        n = len(labels)
        k = int(n * noise_ratio)
        
        if k > 0:
            idx = np.random.choice(n, size=k, replace=False)
            for i in idx:
                old = labels[i]
                new = random.randrange(num_classes)
                while new == old:
                    new = random.randrange(num_classes)
                labels[i] = new
            print(f"[RAF-IR50] Injected noise: {k}/{n} ({noise_ratio*100:.1f}%) labels corrupted")
        
        return labels
    
    def __getitem__(self, index):
        """
        返回一个数据样本
        
        返回:
        - img: Tensor (3, H, W), 归一化到 [-1, 1]
        - label: LongTensor, 表情标签 [0, num_classes-1]
        - landmarks: FloatTensor (136,), 归一化到 [0, 1]
        """
        # 1. 加载图像
        img_path = self.file_paths[index]
        img = Image.open(img_path).convert("RGB")
        
        # 2. 加载标签
        label = int(self.train_labels[index])
        label = torch.tensor(label, dtype=torch.long)
        
        # 3. 加载关键点 (原始像素坐标)
        landmarks = torch.from_numpy(self.landmarks_data[index]).float()
        
        # 4. 应用联合变换 (图像 + 关键点同步)
        if self.joint_transform is not None:
            img, landmarks = self.joint_transform(img, landmarks)
        else:
            # 兼容旧接口：仅变换图像
            img = self.transform(img)
        
        # 5. 关键点归一化到 [0, 1]
        # 此时 landmarks 已经过 resize/crop，处于 resize 后的像素坐标
        # 除以 img_size 即可归一化
        landmarks = landmarks / float(self.args.img_size)
        
        # 确保归一化后的关键点在 [0, 1] 范围内
        landmarks = torch.clamp(landmarks, 0.0, 1.0)
        
        return img, label, landmarks
    
    def __len__(self):
        return len(self.file_paths)


def get_raf_dataloaders_ir50(args, batch_size=32, num_workers=4):
    """
    创建 RAF-DB 的 train 和 val DataLoader (IR50版本)
    
    参数:
    - args: 配置对象
    - batch_size: batch size
    - num_workers: 数据加载线程数
    
    返回:
    - train_loader: 训练集 DataLoader
    - val_loader: 验证集 DataLoader
    """
    train_dataset = RAFDataset_IR50(
        args, 
        phase='train', 
        use_augmentation=True
    )
    
    val_dataset = RAFDataset_IR50(
        args, 
        phase='val', 
        use_augmentation=False
    )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 确保 batch 大小一致
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ========== 兼容性示例：与 MultiTaskModel_IR50 配合使用 ==========

# if __name__ == "__main__":
#     # 示例配置
#     class Args:
#         data_path = "./Dataset/RAF-DB/Image/aligned"
#         raf_train_csv = "./Dataset/RAF-DB/train.csv"
#         raf_val_csv = "./Dataset/RAF-DB/test.csv"
#         img_size = 112  # 或 224
#         num_classes = 7
#         num_landmarks = 68
#         noise_ratio = 0.0  # 无噪声
    
#     args = Args()
    
#     # 创建数据集
#     train_dataset = RAFDataset_IR50(args, phase='train')
#     val_dataset = RAFDataset_IR50(args, phase='val')
    
#     print(f"Train: {len(train_dataset)} samples")
#     print(f"Val:   {len(val_dataset)} samples")
    
#     # 测试一个样本
#     img, label, landmarks = train_dataset[0]
#     print(f"\nSample shape:")
#     print(f"  Image:     {img.shape}")      # (3, 112, 112) or (3, 224, 224)
#     print(f"  Label:     {label}")           # 0-6
#     print(f"  Landmarks: {landmarks.shape}") # (136,)
#     print(f"  Landmarks range: [{landmarks.min():.4f}, {landmarks.max():.4f}]")  # [0, 1]

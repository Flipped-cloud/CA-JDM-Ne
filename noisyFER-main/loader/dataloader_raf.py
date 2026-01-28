import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

# from utils_fusion import joint_transforms
# ...existing code...

class RAFDataset_Fusion(data.Dataset):
    def __init__(self, args, phase, transform=None):
        self.phase = phase
        self.args = args
        self.data_root = args.data_path  # 包含 train.csv/test.csv 和 Image 文件夹的根目录

        # -----------------------------------------------------------
        # 1. 定义图像变换：与 decoder 的 tanh 输出域对齐到 [-1, 1]
        # -----------------------------------------------------------
        if transform is not None:
            self.transform = transform
        else:
            if phase == "train":
                self.transform = T.Compose(
                    [
                        T.Resize((args.img_size, args.img_size)),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),  # [0,1]
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1,1]
                    ]
                )
            else:
                self.transform = T.Compose(
                    [
                        T.Resize((args.img_size, args.img_size)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )

        # -----------------------------------------------------------
        # 2. 从 CSV 加载数据
        # -----------------------------------------------------------
        csv_file = os.path.join(self.data_root, f"{phase}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")

        print(f"Loading {phase} data from {csv_file}...")
        df = pd.read_csv(csv_file)

        # 2.1 解析图像路径
        self.file_paths = []
        for img_id in df["image_id"]:
            img_name = f"{img_id}.jpg" if not str(img_id).endswith(".jpg") else str(img_id)
            img_path = os.path.join(self.data_root, "Image", "aligned", img_name)
            if not os.path.exists(img_path):
                # 兼容一些数据集把图片直接放 Image/ 下
                alt_path = os.path.join(self.data_root, "Image", img_name)
                img_path = alt_path if os.path.exists(alt_path) else img_path
            self.file_paths.append(img_path)

        # 2.2 解析标签（兼容 1-7 或 0-6）
        raw_labels = df["label"].values.astype(np.int64)
        if raw_labels.max() == 7 and raw_labels.min() >= 1:
            clean_labels = raw_labels - 1
        else:
            clean_labels = raw_labels

        self.clean_labels = clean_labels

        # 注入噪声（仅训练集）
        if phase == "train":
            self.train_labels = self.inject_noise(
                self.clean_labels, noise_ratio=args.noise_ratio, num_classes=args.num_classes
            )
        else:
            self.train_labels = self.clean_labels

        # RAF 融合版：如果你的 CSV 含关键点列，可在这里读取；否则用占位零向量
        self.landmarks_data = None
        x_cols = [f"x_{i}" for i in range(args.num_landmarks)]
        y_cols = [f"y_{i}" for i in range(args.num_landmarks)]
        if all(c in df.columns for c in (x_cols + y_cols)):
            lmk = df[x_cols + y_cols].values.astype(np.float32)
            self.landmarks_data = lmk
        else:
            self.landmarks_data = np.zeros((len(self.file_paths), args.num_landmarks * 2), dtype=np.float32)

    def inject_noise(self, labels, noise_ratio, num_classes=7):
        labels = np.asarray(labels, dtype=np.int64).copy()
        n = len(labels)
        k = int(n * noise_ratio)
        idx = np.random.choice(n, size=k, replace=False)
        for i in idx:
            old = labels[i]
            new = random.randrange(num_classes)
            while new == old:
                new = random.randrange(num_classes)
            labels[i] = new
        return labels

    def __getitem__(self, index):
        img_path = self.file_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = int(self.train_labels[index])
        label = torch.tensor(label, dtype=torch.long)

        landmarks = torch.from_numpy(self.landmarks_data[index]).float()
        # 关键点坐标若是像素坐标，归一化到 [0,1]（与你现有训练逻辑一致）
        landmarks = landmarks / float(self.args.img_size)

        return img, label, landmarks

    def __len__(self):
        return len(self.file_paths)
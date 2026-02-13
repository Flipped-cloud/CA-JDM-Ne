import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data

from . import joint_transforms as JT

# from utils_fusion import joint_transforms
# ...existing code...

class RAFDataset_Fusion(data.Dataset):
    def __init__(self, args, phase, transform=None):
        self.phase = phase
        self.args = args
        # NOTE:
        # - args.data_path: image root (e.g. ...\Dataset\RAF-DB\Image\aligned)
        # - args.raf_train_csv / args.raf_val_csv: explicit CSV paths
        self.image_root = args.data_path

        # -----------------------------------------------------------
        # 1. 定义联合变换（图像与关键点同步 Resize / Flip）
        #    说明：RAF 的 train.csv/test.csv 中关键点是“原图像素坐标”。
        #    - 先用 JointTransform 把关键点按原图尺寸 -> resize 后尺寸进行缩放/翻转
        #    - 再把关键点归一化到 [0,1]（除以 args.img_size），与训练逻辑保持一致
        # -----------------------------------------------------------
        self.transform = transform
        if self.transform is None:
            if phase == "train":
                transforms_list = [
                    JT.Resize((args.img_size, args.img_size)),
                    JT.RandomHorizontalFlip(p=0.5),
                ]
                if not getattr(args, "no_color_jitter", False):
                    transforms_list.append(
                        JT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                    )
                transforms_list.extend(
                    [
                        JT.ToTensor(),
                        JT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
                if not getattr(args, "no_random_erasing", False):
                    transforms_list.append(JT.RandomErasing(p=0.5))
                self.joint_transform = JT.Compose(transforms_list)
            else:
                self.joint_transform = JT.Compose(
                    [
                        JT.Resize((args.img_size, args.img_size)),
                        JT.ToTensor(),
                        JT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
        else:
            # 用户显式传入 transform 时：保持旧行为（只做图像变换，不做 landmark 同步增强）。
            self.joint_transform = None

        # -----------------------------------------------------------
        # 2. 从 CSV 加载数据
        # -----------------------------------------------------------
        if phase == "train":
            csv_file = getattr(args, "raf_train_csv", None)
        else:
            csv_file = getattr(args, "raf_val_csv", None)

        # Backward-compatible fallback: look for {phase}.csv next to image_root or its parent.
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

        print(f"Loading {phase} data from {csv_file}...")
        df = pd.read_csv(csv_file)

        # 2.1 解析图像路径
        # CSV image column options: image_id | image | path | file | img | img_path
        image_col = None
        for cand in ("image_id", "image", "path", "file", "img", "img_path", "image_path"):
            if cand in df.columns:
                image_col = cand
                break
        if image_col is None:
            # Fall back to first column if user-provided CSV has no header or unknown header
            image_col = df.columns[0]

        self.file_paths = []
        for v in df[image_col].values:
            s = str(v).strip()
            # If it's a pure id (digits), default to .jpg
            if s.isdigit():
                s = f"{s}.jpg"
            elif not os.path.splitext(s)[1]:
                # no extension provided
                s = f"{s}.jpg"

            # If CSV already contains a relative path (relative to Image\\aligned), join with image_root.
            # If it contains an absolute path, keep it.
            img_path = s if os.path.isabs(s) else os.path.join(self.image_root, s)
            self.file_paths.append(img_path)

        # 2.2 解析标签（兼容 1-7 或 0-6）
        label_col = "label" if "label" in df.columns else None
        if label_col is None:
            # If CSV has two columns without standard names, assume second column is label
            if len(df.columns) >= 2:
                label_col = df.columns[1]
            else:
                raise KeyError(
                    f"Label column not found in CSV: {csv_file}. "
                    f"Expected a 'label' column or at least 2 columns."
                )

        raw_labels = df[label_col].values.astype(np.int64)
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

        # Detect if landmarks are already normalized to [0,1]
        if len(self.landmarks_data) > 0:
            lmk_max = float(np.nanmax(self.landmarks_data))
        else:
            lmk_max = 0.0
        self.landmarks_normalized = lmk_max <= 1.5

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

        label = int(self.train_labels[index])
        label = torch.tensor(label, dtype=torch.long)

        landmarks = torch.from_numpy(self.landmarks_data[index]).float()

        if self.joint_transform is not None:
            if self.landmarks_normalized:
                w, h = img.size
                landmarks = landmarks.clone()
                landmarks[0::2] = landmarks[0::2] * float(w)
                landmarks[1::2] = landmarks[1::2] * float(h)
            img, landmarks = self.joint_transform(img, landmarks)
        else:
            # 兼容旧接口：仅变换图像，不同步变换关键点。
            img = self.transform(img)

        # 关键点在 joint resize 之后处于“resize 后的像素坐标”，再归一化到 [0,1]
        landmarks = landmarks / float(self.args.img_size)

        return img, label, landmarks

    def __len__(self):
        return len(self.file_paths)
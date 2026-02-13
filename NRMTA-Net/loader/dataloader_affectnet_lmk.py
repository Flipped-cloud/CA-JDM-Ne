import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from . import joint_transforms


class AffectNetDataset_Fusion(data.Dataset):
    """AffectNet dataset with landmarks for NRMTA-Net."""

    def __init__(self, args, phase):
        self.phase = phase
        self.args = args
        self.img_root = args.affectnet_img_root
        self.num_classes = args.num_classes
        self.num_landmarks = args.num_landmarks

        # -----------------------------------------------------------
        # 1) 图像变换：与 decoder tanh 输出域对齐到 [-1,1]
        # -----------------------------------------------------------
        if phase == "train":
            self.transform = T.Compose(
                [
                    T.Resize((args.img_size, args.img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            csv_file = args.affectnet_train_csv
        else:
            self.transform = T.Compose(
                [
                    T.Resize((args.img_size, args.img_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            csv_file = args.affectnet_val_csv

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"AffectNet CSV not found: {csv_file}")

        df = pd.read_csv(csv_file)

        x_cols = [f"x_{i}" for i in range(self.num_landmarks)]
        y_cols = [f"y_{i}" for i in range(self.num_landmarks)]
        missing_cols = [c for c in x_cols + y_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing landmark columns in CSV: {missing_cols[:10]} ...")

        self.file_paths = []
        self.labels = []
        self.landmarks_data = []

        for _, row in df.iterrows():
            rel_path = str(row["subDirectory_filePath"]) if "subDirectory_filePath" in df.columns else str(row["img_path"])
            img_path = os.path.join(self.img_root, rel_path)
            self.file_paths.append(img_path)

            lbl = int(row["expression"]) if "expression" in df.columns else int(row["label"])
            self.labels.append(lbl)

            lmk = np.concatenate([row[x_cols].to_numpy(), row[y_cols].to_numpy()]).astype(np.float32)
            self.landmarks_data.append(lmk)

        self.landmarks_data = np.asarray(self.landmarks_data, dtype=np.float32)

        if len(self.file_paths) != len(self.landmarks_data):
            raise RuntimeError("file_paths and landmarks_data length mismatch")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_path = self.file_paths[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}, err={e}")

        img = self.transform(img)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        landmarks = torch.from_numpy(self.landmarks_data[index]).float()
        # 像素坐标 -> [0,1]（与你现有训练逻辑一致）
        landmarks = landmarks / float(self.args.img_size)

        return img, label, landmarks

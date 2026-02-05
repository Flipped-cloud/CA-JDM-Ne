import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from tqdm import tqdm  # 引入进度条，让你看到它在动

class AffectNetDataset_Fusion(data.Dataset):
    """AffectNet dataset with landmarks for CA-JDM-Net."""

    def __init__(self, args, phase):
        self.phase = phase
        self.args = args
        self.img_root = args.affectnet_img_root
        self.num_classes = args.num_classes
        self.num_landmarks = args.num_landmarks
        self.img_size = args.img_size

        # -----------------------------------------------------------
        # 1) 图像变换
        # -----------------------------------------------------------
        if phase == "train":
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            csv_file = args.affectnet_train_csv
        else:
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            csv_file = args.affectnet_val_csv

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"AffectNet CSV not found: {csv_file}")

        print(f"[{phase}] Loading CSV: {csv_file} ...")
        df = pd.read_csv(csv_file)

        # -----------------------------------------------------------
        # 2) 极速处理：列名识别 & 路径处理
        # -----------------------------------------------------------
        # 确定路径列名
        if "subDirectory_filePath" in df.columns:
            path_col = "subDirectory_filePath"
        elif "img_path" in df.columns:
            path_col = "img_path"
        else:
            path_col = "image_id" # 你的CSV列名

        # 确定标签列名
        if "emotion" in df.columns:
            label_col = "emotion"
        elif "expression" in df.columns:
            label_col = "expression"
        else:
            label_col = "label"

        # -----------------------------------------------------------
        # 3) 极速处理：向量化读取 (瞬间完成)
        # -----------------------------------------------------------
        # (A) 路径：直接把文件名拼接到 root 后面
        # 使用列表推导式，比 iterrows 快得多
        print(f"[{phase}] Parsing paths...")
        raw_paths = df[path_col].astype(str).tolist()
        self.file_paths = [os.path.join(self.img_root, p) for p in raw_paths]

        # (B) 标签：批量处理复杂标签
        print(f"[{phase}] Parsing labels...")
        raw_labels = df[label_col].tolist()
        
        # 定义解析函数
        label_map = {
            "Surprise": 0, "Fear": 1, "Disgust": 2, "Happiness": 3,
            "Sadness": 4, "Anger": 5, "Neutral": 6, "Contempt": 7
        }
        
        def parse_label(val):
            try:
                return int(val) # 大部分情况直接转数字
            except ValueError:
                val_str = str(val).strip()
                # 尝试 "3 Happiness" -> 3
                first_part = val_str.split(' ')[0]
                if first_part.isdigit():
                    return int(first_part)
                # 尝试 "Happiness" -> 3
                for k, v in label_map.items():
                    if k in val_str:
                        return v
                # 无法识别
                return 6 # 默认 Neutral

        # 使用 map 快速应用函数
        self.labels = [parse_label(l) for l in raw_labels]

        # (C) 关键点：直接读取数值矩阵 (最快)
        print(f"[{phase}] Parsing landmarks...")
        x_cols = [f"x_{i}" for i in range(self.num_landmarks)]
        y_cols = [f"y_{i}" for i in range(self.num_landmarks)]
        
        # 检查列是否存在
        if x_cols[0] not in df.columns:
            # 如果没有关键点列，填充0
            print("Warning: Landmark columns not found! Filling with zeros.")
            self.landmarks_data = np.zeros((len(df), self.num_landmarks * 2), dtype=np.float32)
        else:
            # Pandas values 直接转 numpy，速度极快
            # data shape: [N, 68]
            lmk_x = df[x_cols].values.astype(np.float32)
            lmk_y = df[y_cols].values.astype(np.float32)
            self.landmarks_data = np.concatenate([lmk_x, lmk_y], axis=1)

        print(f"[{phase}] Loaded {len(self.file_paths)} images.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_path = self.file_paths[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (self.img_size, self.img_size))
        
        w, h = img.size 
        img_tensor = self.transform(img)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # 关键点归一化
        raw_lmk = self.landmarks_data[index]
        num_pts = self.num_landmarks
        
        xs = raw_lmk[:num_pts]
        ys = raw_lmk[num_pts:]

        # 加上 max(1) 防止除以0
        w = max(w, 1)
        h = max(h, 1)

        xs_norm = xs / w
        ys_norm = ys / h
        
        landmarks = np.concatenate([xs_norm, ys_norm])
        landmarks = torch.from_numpy(landmarks).float()

        return img_tensor, label, landmarks
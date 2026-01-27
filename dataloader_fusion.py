import torch.utils.data as data
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os
import random
# from utils_fusion import joint_transforms
import joint_transforms 

class RAFDataset_Fusion(data.Dataset):
    def __init__(self, args, phase, transform=None):
        self.phase = phase
        self.args = args
        self.data_root = args.data_path  # 包含 train.csv/test.csv 和 Image 文件夹的根目录
        
        # -----------------------------------------------------------
        # 1. 定义联合变换 (Joint Transforms) - 保持不变
        # -----------------------------------------------------------
        if phase == 'train':
            self.transform = joint_transforms.Compose([
                joint_transforms.Resize((224, 224)),
                # 训练时开启随机翻转，增强抗噪能力
                joint_transforms.RandomHorizontalFlip(p=0.5), 
                joint_transforms.ToTensor(),
                joint_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = joint_transforms.Compose([
                joint_transforms.Resize((224, 224)),
                joint_transforms.ToTensor(),
                joint_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])

        # -----------------------------------------------------------
        # 2. 从 CSV 加载数据 (核心修改)
        # -----------------------------------------------------------
        csv_file = os.path.join(self.data_root, f'{phase}.csv')
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}. Please ensure train.csv/test.csv are in {self.data_root}")
            
        print(f"Loading {phase} data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # 2.1 解析图像路径
        # 假设 CSV 中的 image_id 是 'train_12270_aligned'，实际文件可能在 Image/aligned/ 下，且需要加上 .jpg
        # 这里假设图片存放结构为: data_root/Image/aligned/train_12270_aligned.jpg
        self.file_paths = []
        for img_id in df['image_id']:
            img_name = str(img_id)
            if not img_name.endswith('.jpg'):
                img_name += '.jpg'
            # 路径拼接：根目录 -> Image -> aligned -> 图片名
            # 如果您的图片直接放在根目录，请修改这里
            path = os.path.join(self.data_root, 'Image', 'aligned', img_name)
            self.file_paths.append(path)
            
        # 2.2 解析标签
        # 注意: 您的CSV中 label 是 6 对应 Neutral。RAF-DB 标准是 1-7 (7=Neutral)。
        # 如果您的 label 已经是 0-6 格式 (即 6=Neutral)，则直接使用。
        # 为了保险，我们假设输入是 1-7，如果发现有 7，则全体 -1。
        # 如果输入已经是 0-6，则保持不变。
        raw_labels = df['label'].values
        if raw_labels.max() == 7 and raw_labels.min() >= 1:
            print("Detected 1-based labels (1-7), converting to 0-based (0-6).")
            self.clean_labels = [l - 1 for l in raw_labels]
        else:
            self.clean_labels = raw_labels.tolist()

        # 2.3 解析关键点 (从 columns x_0...x_67, y_0...y_67)
        # 我们需要格式: [x0, y0, x1, y1, ... x67, y67]
        self.landmarks_data = []
        
        # 提取列名
        x_cols = [f'x_{i}' for i in range(68)]
        y_cols = [f'y_{i}' for i in range(68)]
        
        # 批量提取并转为 Numpy (速度极快)
        X = df[x_cols].values # [N, 68]
        Y = df[y_cols].values # [N, 68]
        
        # 组合为交错格式 [x0, y0, x1, y1...]
        # stack: [N, 68, 2] -> flatten -> [N, 136]
        L = np.stack((X, Y), axis=2).reshape(-1, 136)
        self.landmarks_data = L.astype(np.float32)

        print(f"Loaded {len(self.file_paths)} samples.")

        # -----------------------------------------------------------
        # 3. 噪声标签注入 (保持 noisyFER 逻辑)
        # -----------------------------------------------------------
        if phase == 'train' and args.noise_ratio > 0:
            print(f"Injecting {args.noise_ratio*100}% symmetric noise...")
            self.train_labels = self.inject_noise(self.clean_labels, args.noise_ratio, num_classes=7)
        else:
            self.train_labels = self.clean_labels

    def inject_noise(self, labels, noise_ratio, num_classes=7):
        """noisyFER 核心: 生成带噪声的标签分布"""
        noisy_labels = []
        for label in labels:
            if random.random() < noise_ratio:
                possible = list(range(num_classes))
                # 这里的 label 已经是 int 了
                if label in possible:
                    possible.remove(label)
                noisy_labels.append(random.choice(possible))
            else:
                noisy_labels.append(label)
        return noisy_labels

    def __getitem__(self, index):
        # 1. 读取图像
        img_path = self.file_paths[index]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # 这里的 image_id 是从 CSV 读的，如果找不到图片，说明路径配置不对或者图片缺失
            # 生成一张黑图防止崩溃，并打印警告（仅调试用）
            # print(f"Warning: Image not found {img_path}") 
            image = Image.new('RGB', (100, 100))
        
        # 2. 获取 Label
        label = self.train_labels[index]
        
        # 3. 获取 Landmarks (直接从内存读取，无需 IO)
        landmarks = self.landmarks_data[index] # Numpy array [136]
        landmarks = torch.from_numpy(landmarks).float()
        
        # 4. 联合变换 (关键点翻转 + Resize)
        if self.transform is not None:
            image, landmarks = self.transform(image, landmarks)
            
        # 5. 归一化关键点 [0, 1]
        landmarks = landmarks / 224.0
        
        return image, label, landmarks, index

    def __len__(self):
        return len(self.file_paths)
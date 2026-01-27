import sys
import os
import torch
import torch.utils.data as data
import cv2
import pandas as pd
import numpy as np
from PIL import Image

# 获取项目根目录
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

# --- 保留原有的标签映射逻辑 ---
# RAF: 1: surprise, 2: fear, 3: disgust, 4: happiness, 5: sadness, 6: anger, 7: neutral
# Affectnet: 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger
transition_to_affectnet = {
    1: 3, 2: 4, 3: 5, 4: 1, 5: 2, 6: 6, 7: 0
}

class DataloaderRAF(data.Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        参数:
            data_dir (str): 数据集根目录 (包含 train_label.csv 的文件夹)
            split (str): 'train' 或 'test'
            transform:虽然保留接口，但建议在内部使用 __ImagePro 处理 Resize
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.mode = 'Train' if split == 'train' else 'Test'

        # 1. 读取 CSV 文件 (替代原来的 txt 读取)
        # 请确保你的 data_dir 下有 train_label.csv 和 test_label.csv (包含关键点)
        csv_name = 'train_label.csv' if self.split == 'train' else 'test_label.csv'
        self.df = pd.read_csv(os.path.join(self.data_dir, csv_name))

        # 2. 定义关键点列名 (自动生成 x_0...x_67, y_0...y_67)
        self.num_landmarks = 68
        self.lmks = []
        for i in range(self.num_landmarks):
            self.lmks.append('x_' + str(i))
            self.lmks.append('y_' + str(i))

        print(f'[{split}] Preprocessing completed, find {len(self.df)} useful images')

    def __ImagePro(self, img, landmarks):
        """
        【新增核心函数】同时处理图片和关键点的缩放
        """
        target_size = 224 # ResNet 输入尺寸，根据需要调整
        w, h = img.size
        
        # 图片缩放
        img = img.resize((target_size, target_size), Image.BILINEAR)
        
        # 关键点缩放
        scale_x = target_size / w
        scale_y = target_size / h
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
        
        return img, landmarks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 1. 构造图片路径
        # 原逻辑：img_name[0:-4] + "_aligned" + img_name[-4:]
        # 这里假设 CSV 中的 name 已经是文件名，如果需要加 _aligned 请保留原逻辑
        raw_name = str(self.df.loc[index, 'name']) # 假设CSV列名叫 name
        
        # 兼容原来的路径逻辑 (如果有 _aligned 文件夹结构)
        if '_aligned' not in raw_name:
            file_name = raw_name[:-4] + "_aligned" + raw_name[-4:]
        else:
            file_name = raw_name
            
        # 拼接完整路径 (根据你的文件夹结构调整中间的 'basic/Image/aligned')
        img_path = os.path.join(self.data_dir, 'basic/Image/aligned', file_name)
        
        # 如果找不到文件，尝试直接路径
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, raw_name)

        # 2. 读取图片 (使用 PIL 以配合 transforms)
        image = Image.open(img_path).convert('RGB')

        # 3. 读取并处理标签 (应用映射)
        original_lbl = int(self.df.loc[index, 'label'])
        # 如果标签是 1-7 (RAF格式)，转为 AffectNet 格式
        # 如果已经是 0-6，则不需要转换，请根据实际 CSV 检查
        label = transition_to_affectnet.get(original_lbl, original_lbl - 1) 

        # 4. 读取关键点 (从 CSV)
        landmarks = torch.Tensor(list(self.df.loc[index, self.lmks]))
        # 重塑为 (68, 2) - 假设 CSV 是 x0...x67, y0...y67 排列
        # 如果是 x0,y0,x1,y1... 排列，用 landmarks.view(-1, 2)
        landmarks = torch.cat([
            landmarks[:self.num_landmarks].view(-1, 1), 
            landmarks[self.num_landmarks:].view(-1, 1)
        ], dim=1)

        # 5. 同步变换 (Resize)
        # 这一步替代了 self.transform 里的 Resize，防止关键点错位
        image, landmarks = self.__ImagePro(image, landmarks)

        # 6. 转 Tensor 并归一化
        # 手动执行 ToTensor 和 Normalize，模拟 noisyFER 的预处理
        # noisyFER 原版: ((img / 255.0 - 0.5) / 0.5)
        import torchvision.transforms.functional as TF
        image = TF.to_tensor(image) # [0, 1]
        image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]

        # 7. 关键点归一化 [0, 1]
        landmarks[:, 0] /= 224
        landmarks[:, 1] /= 224

        # 返回字典 (适配 CA-JDM-Net)
        return {
            'data': image,
            'labels': label,
            'landmarks': landmarks,
            'image_path': img_path # 调试用
        }
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np

# ==========================================
# 1. 配置模块 (Configuration)
# 结合了 cmcnn 的 config.py 和 noisyFER 的参数设置
# ==========================================
class Config:
    def __init__(self):
        # 基础路径配置 (根据实际环境修改)
        self.data_root = './datasets/RAF-DB'  # 假设使用 RAF-DB 作为主要实验数据
        self.train_list_path = os.path.join(self.data_root, 'train_list.txt')
        self.test_list_path = os.path.join(self.data_root, 'test_list.txt')
        
        # 图像参数
        self.img_size = 224
        self.crop_size = 224
        
        # 训练超参数 (Day 1 基线设置)
        self.batch_size = 32  # 调试阶段使用较小 batch
        self.workers = 4
        self.lr = 0.001
        self.epochs = 50
        self.seed = 42
        
        # 噪声标签相关 (来自 noisyFER)
        self.noise_rate = 0.0  # Day 1 可能先跑 Clean Baseline，后续增加噪声
        self.noise_type = 'symmetric' # 'symmetric' or 'asymmetric'

    def print_config(self):
        print("====== Experiment Configuration (Day 1) ======")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("==============================================")

# ==========================================
# 2. 数据增强与预处理 (Transforms)
# 参考 noisyFER-main/transforms.py
# ==========================================
def get_transforms(cfg, split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((cfg.crop_size, cfg.crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 增加色彩扰动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((cfg.crop_size, cfg.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# ==========================================
# 3. 数据集类 (Dataset)
# 整合 RAF-DB/AffectNet 读取逻辑
# ==========================================
class FERDataset(Dataset):
    def __init__(self, cfg, split='train', transform=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.samples = []
        
        # 模拟数据加载逻辑 (根据实际 list 文件格式调整)
        # 这里假设格式为: "image_path label"
        # 如果是真实运行，需要确保路径文件存在
        
        # Day 1: 这里我们创建一个 Mock 数据生成逻辑，方便直接运行测试 pipeline
        # 在实际部署时，将 use_mock_data 设为 False 并提供真实路径
        self.use_mock_data = True 
        
        if not self.use_mock_data:
            list_path = cfg.train_list_path if split == 'train' else cfg.test_list_path
            with open(list_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    path, label = line.strip().split()
                    self.samples.append((os.path.join(cfg.data_root, 'Image/aligned', path), int(label)))
        else:
            print(f"[Info] Using MOCK data for {split} set (Day 1 check).")
            for i in range(100): # 模拟100张图
                self.samples.append((f"mock_img_{i}.jpg", np.random.randint(0, 7)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if self.use_mock_data:
            # 生成随机噪声图作为模拟
            img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255)).convert('RGB')
        else:
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # 返回一张黑色图防止崩坏
                img = Image.new('RGB', (self.cfg.img_size, self.cfg.img_size))

        if self.transform:
            img = self.transform(img)
            
        return img, label, idx  # 返回 idx 用于后续可能的噪声标签处理

# ==========================================
# 4. 主流程验证 (Main Execution)
# ==========================================
def main():
    print(">>> Starting Day 1: Data Pipeline & Environment Setup...")
    
    # 1. 初始化配置
    cfg = Config()
    cfg.print_config()
    
    # 2. 检查 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # 3. 准备数据加载器
    train_transform = get_transforms(cfg, 'train')
    val_transform = get_transforms(cfg, 'val')
    
    train_dataset = FERDataset(cfg, split='train', transform=train_transform)
    val_dataset = FERDataset(cfg, split='test', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.workers
    )
    
    print(f">>> Data Loaders Ready.")
    print(f"    Train Samples: {len(train_dataset)}")
    print(f"    Val Samples:   {len(val_dataset)}")
    
    # 4. 测试 Batch 读取
    print(">>> Testing Batch Loading...")
    try:
        for i, (images, labels, indices) in enumerate(train_loader):
            print(f"    Batch {i}: Image Shape {images.shape}, Labels Shape {labels.shape}")
            if i >= 2: break # 只测试前3个batch
            
        print(">>> Day 1 Tasks Completed: Data Pipeline Verified Successfully.")
        
    except Exception as e:
        print(f"!!! Error in Data Pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    main()
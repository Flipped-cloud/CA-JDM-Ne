import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader_raf import RAFDataset_Fusion

# 设置中文字体 (防止中文乱码，根据系统自动适配)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def denormalize(image, landmarks):
    """反归一化，用于可视化"""
    # ImageNet Mean/Std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Image: (3, H, W) -> (H, W, 3)
    img = image.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Landmarks: [0, 1] -> [0, 224]
    lmks = landmarks.numpy() * 224
    lmks = lmks.reshape(-1, 2)
    return img, lmks

def visualize_batch(dataloader):
    print("正在抽取样本进行可视化...")
    try:
        # 获取一个 Batch
        batch = next(iter(dataloader))
        images, labels, landmarks, indices = batch
    except Exception as e:
        print(f"数据读取失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4x2 网格显示
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle('Dataloader Verification (RAF-DB)', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        
        img_np, lmk_np = denormalize(images[i], landmarks[i])
        label = labels[i].item()
        
        ax.imshow(img_np)
        
        # 画所有绿点 (验证 CSV 坐标是否准确)
        ax.scatter(lmk_np[:, 0], lmk_np[:, 1], s=4, c='g', alpha=0.7)
        
        # 验证翻转逻辑 (左眼红，右眼蓝)
        # 索引基于 standard 68 points:
        # 36: 左眼外角 -> 视觉左边 (未翻转时)
        # 45: 右眼外角 -> 视觉右边 (未翻转时)
        # 如果图片翻转了，红点应该依然跟随逻辑上的“左眼”(现在在视觉右边)，或者根据你的翻转逻辑互换
        ax.scatter(lmk_np[36, 0], lmk_np[36, 1], s=40, c='r', label='Left Eye(36)')
        ax.scatter(lmk_np[45, 0], lmk_np[45, 1], s=40, c='b', label='Right Eye(45)')
        
        ax.set_title(f"Label: {label}\nIdx:{indices[i].item()}", fontsize=10)
        ax.axis('off')

    # 只在第一个图显示图例
    axes[0,0].legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    # 保存结果
    save_path = 'verify_result.png'
    plt.savefig(save_path)
    print(f"可视化结果已保存至: {os.path.abspath(save_path)}")
    # plt.show() # 如果在服务器运行，请注释掉这行

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认路径设为当前目录，方便测试
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root (containing train.csv)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    
    args = parser.parse_args()

    print(f"Checking data path: {args.data_path}")
    
    if not os.path.exists(args.data_path):
        print(f"Error: Path not found {args.data_path}")
        exit()

    try:
        # 初始化
        dataset = RAFDataset_Fusion(args, phase='train')
        
        if len(dataset) == 0:
            print("错误: 数据集为空。请检查 CSV 文件路径和图片路径配置。")
        else:
            print(f"数据集加载成功，共 {len(dataset)} 张图片。")
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            visualize_batch(dataloader)
            
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
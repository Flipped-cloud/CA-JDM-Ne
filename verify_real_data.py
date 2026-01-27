import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#from loader.dataloader_fusion import RAFDataset_Fusion
from dataloader_fusion import RAFDataset_Fusion

# 设置中文字体 (Windows通常为SimHei, Mac为Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def denormalize(image, landmarks):
    # ImageNet Mean/Std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Landmarks: 0-1 -> 0-224
    lmks = landmarks.numpy() * 224
    lmks = lmks.reshape(-1, 2)
    return img, lmks

def visualize_batch(dataloader):
    print("正在抽取样本进行可视化...")
    try:
        # 获取一个 Batch
        images, labels, landmarks, indices = next(iter(dataloader))
    except Exception as e:
        print(f"数据读取失败: {e}")
        return

    # 4x2 网格显示
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle('CSV 数据加载与变换测试 (Day 1)', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        
        img_np, lmk_np = denormalize(images[i], landmarks[i])
        label = labels[i].item()
        
        ax.imshow(img_np)
        
        # 画所有绿点 (验证 CSV 坐标是否准确)
        ax.scatter(lmk_np[:, 0], lmk_np[:, 1], s=4, c='g', alpha=0.7)
        
        # 验证翻转逻辑 (左眼红，右眼蓝)
        # 如果图片被翻转，红点依然应该在视觉上的左边 (即逻辑上的左眼被交换到了左边)
        ax.scatter(lmk_np[36, 0], lmk_np[36, 1], s=40, c='r', label='Idx36(左)')
        ax.scatter(lmk_np[45, 0], lmk_np[45, 1], s=40, c='b', label='Idx45(右)')
        
        ax.set_title(f"Label: {label}\nIdx:{indices[i].item()}", fontsize=10)
        ax.axis('off')

    # 只在第一个图显示图例，避免遮挡
    axes[0,0].legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 必须指向包含 train.csv 和 Image 文件夹的根目录
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.data_path, 'train.csv')):
        print(f"错误：在 {args.data_path} 下未找到 train.csv")
        exit()

    # 初始化 (读取 train.csv)
    dataset = RAFDataset_Fusion(args, phase='train')
    
    # 检查是否找到了图片
    # 如果 CSV 有数据但 file_paths 指向的文件不存在，需要根据错误调整路径逻辑
    first_img_path = dataset.file_paths[0]
    if not os.path.exists(first_img_path):
        print(f"\n[警告] CSV加载成功，但检测到图片路径不存在！")
        print(f"期望路径: {first_img_path}")
        print("请检查 loader/dataloader_fusion.py 中的路径拼接逻辑 (Lines 49-55)")
        # 不退出，让程序继续运行以便抛出更详细的 Error
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    visualize_batch(dataloader)
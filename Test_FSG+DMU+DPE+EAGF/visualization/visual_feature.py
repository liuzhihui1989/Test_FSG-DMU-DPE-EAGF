import os
import torch
import matplotlib.pyplot as plt


def visualize_feature_map(feature_map, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu()

    # 默认取第一张图
    fmap = feature_map[0]

    # 对多通道特征图，做通道平均
    if fmap.ndim == 3:
        fmap = fmap.mean(0)

    # 归一化到 0-1
    fmap -= fmap.min()
    fmap /= (fmap.max() + 1e-6)

    # 保存图像
    plt.imshow(fmap, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

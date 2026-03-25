"""
Utility functions module
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_directory(path):
    """
    Create directory (if not exists)
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_config_to_json(config, output_path):
    """
    Save configuration as JSON file
    
    Args:
        config: Configuration object
        output_path: Output path
    """
    config_dict = {
        key: value for key, value in vars(config).items()
        if not key.startswith('_') and not callable(value)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Configuration saved: {output_path}")


def calculate_model_size(model):
    """
    Calculate model size
    
    Args:
        model: Keras model
        
    Returns:
        Model parameter count and size (MB)
    """
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    # Assume float32, 4 bytes per parameter
    size_mb = (total_params * 4) / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'size_mb': size_mb
    }


def compare_images(img1, img2, titles=['Image 1', 'Image 2'], save_path=None):
    """
    Compare two images
    
    Args:
        img1: First image
        img2: Second image
        titles: Title list
        save_path: Save path (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display two images
    im1 = axes[0].imshow(img1, cmap='viridis')
    axes[0].set_title(titles[0], fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(img2, cmap='viridis')
    axes[1].set_title(titles[1], fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Display difference map
    diff = np.abs(img1 - img2)
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_statistics(data, name='Data'):
    """
    计算数据统计信息
    
    Args:
        data: 数据数组
        name: 数据名称
        
    Returns:
        统计信息字典
    """
    stats = {
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'shape': data.shape
    }
    
    print(f"\n{name} 统计信息:")
    print(f"  形状: {stats['shape']}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  平均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    
    return stats


def normalize_data(data, method='minmax', data_range=(0, 255)):
    """
    数据归一化
    
    Args:
        data: 输入数据
        method: 归一化方法 ('minmax', 'zscore')
        data_range: 目标范围（仅用于minmax）
        
    Returns:
        归一化后的数据
    """
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        normalized = normalized * (data_range[1] - data_range[0]) + data_range[0]
    
    elif method == 'zscore':
        mean_val = np.mean(data)
        std_val = np.std(data)
        normalized = (data - mean_val) / (std_val + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def plot_loss_components(history, save_path=None):
    """
    绘制各项损失的详细图表
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 重建损失
    if 'recon_loss' in history.history:
        axes[0, 0].plot(history.history['recon_loss'], linewidth=2)
        axes[0, 0].set_title('Reconstruction Loss', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 梯度一致性损失
    if 'grad_loss' in history.history:
        axes[0, 1].plot(history.history['grad_loss'], linewidth=2, color='green')
        axes[0, 1].set_title('Gradient Consistency Loss', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 平滑性损失
    if 'smooth_loss' in history.history:
        axes[1, 0].plot(history.history['smooth_loss'], linewidth=2, color='orange')
        axes[1, 0].set_title('Smoothness Loss', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 水面连续性损失
    if 'surface_loss' in history.history:
        axes[1, 1].plot(history.history['surface_loss'], linewidth=2, color='purple')
        axes[1, 1].set_title('Surface Continuity Loss', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失组件图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def set_random_seed(seed=42):
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"随机种子设置为: {seed}")


def check_gpu_availability():
    """
    检查GPU可用性
    
    Returns:
        GPU信息字典
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        'available': len(gpus) > 0,
        'count': len(gpus),
        'names': []
    }
    
    if gpus:
        print(f"\n找到 {len(gpus)} 个GPU:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            info['names'].append(gpu.name)
        
        # 获取GPU内存信息
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            info['details'] = gpu_details
        except:
            pass
    else:
        print("\n未找到GPU，将使用CPU训练")
    
    return info


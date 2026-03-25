"""
Main training script
Use refactored modular code for training
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt

import argparse
from config import Config
from data_loader import DataLoader
from trainer import Trainer


def print_system_info():
    """Print system information"""
    print("="*60)
    print("系统信息")
    print("="*60)
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"可用GPU数量: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    print("="*60)


def plot_training_curves(history, save_path='out_save/training_curves.png'):
    """
    Plot training curves
    
    Args:
        history: Training history object
        save_path: Save path
    """
    fig = plt.figure(figsize=(16, 12))
    scale_txt = f"{Config.UPSCALE_FACTOR}x SR"
    
    # 1. Total loss curve
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='train total loss', linewidth=2, color='red')
    plt.plot(history.history['val_loss'], label='val total loss', linewidth=2, linestyle='--', color='red')
    plt.xlabel('Epochs', size=12)
    plt.ylabel('Total Loss', size=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Total Loss ({scale_txt})")
    
    # 2. Individual loss curves
    plt.subplot(2, 2, 2)
    plt.plot(history.history.get('recon_loss', []), label='recon loss', linewidth=2, color='blue')
    plt.plot(history.history.get('grad_loss', []), label='grad loss', linewidth=2, color='green')
    plt.plot(history.history.get('smooth_loss', []), label='smooth loss', linewidth=2, color='orange')
    plt.plot(history.history.get('surface_loss', []), label='surface loss', linewidth=2, color='purple')
    plt.xlabel('Epochs', size=12)
    plt.ylabel('Component Loss', size=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Component Losses ({scale_txt})")
    plt.yscale('log')
    
    # 3. PSNR曲线
    plt.subplot(2, 2, 3)
    plt.plot(history.history['PSNR'], label='train PSNR', linewidth=2, color='darkgreen')
    plt.plot(history.history['val_PSNR'], label='val PSNR', linewidth=2, linestyle='--', color='darkgreen')
    plt.xlabel('Epochs', size=12)
    plt.ylabel('PSNR (dB)', size=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"PSNR ({scale_txt})")
    
    # 4. SSIM曲线
    plt.subplot(2, 2, 4)
    plt.plot(history.history['SSIM'], label='train SSIM', linewidth=2, color='darkblue')
    plt.plot(history.history['val_SSIM'], label='val SSIM', linewidth=2, linestyle='--', color='darkblue')
    plt.xlabel('Epochs', size=12)
    plt.ylabel('SSIM', size=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"SSIM ({scale_txt})")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved: {save_path}")
    plt.close()


def print_final_statistics(history):
    """Print final statistics"""
    print("\n" + "="*60)
    print("Training Statistics (Last Epoch)")
    print("="*60)
    
    if 'recon_loss' in history.history:
        print(f"Reconstruction loss: {history.history['recon_loss'][-1]:.4f}")
    if 'grad_loss' in history.history:
        print(f"Gradient loss: {history.history['grad_loss'][-1]:.4f}")
    if 'smooth_loss' in history.history:
        print(f"Smoothness loss: {history.history['smooth_loss'][-1]:.6f}")
    if 'surface_loss' in history.history:
        print(f"Surface loss: {history.history['surface_loss'][-1]:.4f}")
    if 'loss' in history.history:
        print(f"总损失: {history.history['loss'][-1]:.4f}")
    
    print("-"*60)
    
    if 'PSNR' in history.history:
        print(f"训练PSNR: {history.history['PSNR'][-1]:.2f} dB")
    if 'val_PSNR' in history.history:
        print(f"验证PSNR: {history.history['val_PSNR'][-1]:.2f} dB")
    if 'SSIM' in history.history:
        print(f"训练SSIM: {history.history['SSIM'][-1]:.4f}")
    if 'val_SSIM' in history.history:
        print(f"验证SSIM: {history.history['val_SSIM'][-1]:.4f}")
    
    print("="*60)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='双流EDSR模型训练脚本')
    parser.add_argument('--ablation_mode', type=str, default=None,
                        help='消融实验模式')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    args = parser.parse_args()
    
    # 如果指定了消融模式，覆盖配置
    if args.ablation_mode:
        print(f"\n[Override] 设置消融实验模式为: {args.ablation_mode}")
        Config.ABLATION_MODE = args.ablation_mode
        
        # 自动更新保存路径
        suffix = args.ablation_mode
        Config.MODEL_SAVE_PATH = f"out_save/DDS_SR_best_4x_{suffix}.h5"
        Config.FINAL_MODEL_PATH = f"out_save/DDS_SR_final_4x_{suffix}.h5"
        Config.HISTORY_SAVE_PATH = f"out_save/training_history_DDS_SR_4x_{suffix}.pkl"
        Config.OUTPUT_PRED_FOLDER = f"DDS_SR_test_best_4x_{suffix}/"
        
    if args.epochs:
        Config.TOTAL_EPOCHS = args.epochs

    # 打印系统信息
    print_system_info()
    
    # 打印配置
    Config.print_config()
    
    # 创建数据加载器
    print("\n加载数据...")
    data_loader = DataLoader(Config)
    
    # 创建训练和验证数据集
    train_dataset = data_loader.create_dataset(mode='train')
    val_dataset = data_loader.create_dataset(mode='val')
    
    # 准备数据集
    train_ds = data_loader.prepare_dataset(train_dataset, mode='train')
    val_ds = data_loader.prepare_dataset(val_dataset, mode='val')
    
    # 获取文件列表信息
    train_lr_files, train_hr_files, _ = data_loader.get_file_lists(mode='train')
    val_lr_files, val_hr_files, _ = data_loader.get_file_lists(mode='val')
    
    print(f"训练样本数: {len(train_lr_files)}")
    print(f"验证样本数: {len(val_lr_files)}")
    
    # 检查数据形状
    example_sample = next(iter(train_ds.take(1)))
    print(f"\n数据形状:")
    print(f"  低分辨率水深: {example_sample[0][0].shape}")
    print(f"  高分辨率DEM: {example_sample[0][1].shape}")
    print(f"  高分辨率水深(标签): {example_sample[1].shape}")
    
    # 创建训练器
    trainer = Trainer(Config)
    
    # 构建模型
    model = trainer.build_model()
    model.summary()
    
    # 编译模型
    trainer.compile_model()
    
    # 训练模型
    history = trainer.train(train_ds, val_ds)
    
    # 保存模型和历史
    trainer.save_model()
    trainer.save_history()
    
    # 绘制训练曲线
    plot_training_curves(history)
    
    # 打印最终统计
    print_final_statistics(history)
    
    print("\n✓ 训练完成！")


if __name__ == "__main__":
    main()


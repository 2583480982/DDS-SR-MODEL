"""
Inference script
Use trained model for prediction
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import rasterio
import matplotlib.pyplot as plt
from config import Config
from data_loader import DataLoader
from model import DualStreamEDSR


class Inferencer:
    """Inference class"""
    
    def __init__(self, model_path, config=None):
        """
        Initialize inference
        
        Args:
            model_path: Model path
            config: Configuration object
        """
        self.config = config or Config
        self.model_path = model_path
        
        # Load model
        print(f"Loading model: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Provide custom objects to load model correctly
        custom_objects = {
            'DualStreamEDSR': DualStreamEDSR
        }
        
        # Load model directly (not through Trainer to avoid circular dependency)
        from tensorflow import keras
        self.model = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        print("Model loading completed!")
        
        # Create output directory
        os.makedirs(self.config.OUTPUT_PRED_FOLDER, exist_ok=True)
    
    def predict_single(self, lr_water_path, hr_dem_path, output_path=None, visualize=True):
        """
        Predict for single file
        
        Args:
            lr_water_path: Low-resolution water depth file path
            hr_dem_path: High-resolution DEM file path
            output_path: Output file path (optional)
            visualize: Whether to visualize results
            
        Returns:
            Predicted high-resolution water depth
        """
        # Load data
        print(f"\nProcessing file: {os.path.basename(lr_water_path)}")
        
        # Read data and metadata
        with rasterio.open(lr_water_path) as src:
            lr_water = src.read(1).astype(np.float32)
            lr_metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'width': src.width,
                'height': src.height,
                'dtype': 'float32'
            }
        
        with rasterio.open(hr_dem_path) as src:
            hr_dem = src.read(1).astype(np.float32)
            hr_dem = (hr_dem - np.min(hr_dem)) / (np.max(hr_dem) - np.min(hr_dem) + 1e-8) * 255.0
        
        # Preprocessing
        lr_water = np.expand_dims(lr_water, axis=(0, -1))  # [1, H, W, 1]
        hr_dem = np.expand_dims(hr_dem, axis=(0, -1))      # [1, H*4, W*4, 1]
        
        # Prediction
        print("Inference in progress...")
        # Call model directly to avoid format issues with predict_step
        # Convert to tensor and call model's __call__ method
        lr_water_tensor = tf.constant(lr_water, dtype=tf.float32)
        hr_dem_tensor = tf.constant(hr_dem, dtype=tf.float32)
        
        # Call model directly (not using predict method to avoid predict_step issues)
        hr_pred = self.model([lr_water_tensor, hr_dem_tensor], training=False)
        hr_pred = hr_pred.numpy()  # Convert to numpy array
        hr_pred = np.squeeze(hr_pred)  # 移除batch和channel维度
        hr_pred = np.clip(hr_pred, 0, 255)  # 确保值在合理范围内
        
        # 保存结果
        if output_path is None:
            basename = os.path.basename(lr_water_path).replace('.tif', '.tif')
            output_path = os.path.join(self.config.OUTPUT_PRED_FOLDER, basename)
        
        self.save_prediction(hr_pred, output_path, lr_metadata)
        
        # 可视化
        if visualize:
            self.visualize_result(lr_water[0, :, :, 0], hr_dem[0, :, :, 0], hr_pred, output_path)
        
        print(f"✓ 预测完成，结果保存至: {output_path}")
        return hr_pred
    
    def save_prediction(self, prediction, output_path, lr_metadata):
        """
        保存预测结果为GeoTIFF
        
        Args:
            prediction: 预测结果
            output_path: 输出路径
            lr_metadata: 低分辨率元数据
        """
        if self.config.SAVE_AS_GEOTIFF:
            # 计算高分辨率的transform（使用配置中的上采样倍数）
            scale_factor = self.config.UPSCALE_FACTOR
            lr_transform = lr_metadata['transform']
            hr_transform = rasterio.Affine(
                lr_transform.a / scale_factor,  # x方向分辨率
                lr_transform.b,
                lr_transform.c,
                lr_transform.d,
                lr_transform.e / scale_factor,  # y方向分辨率
                lr_transform.f
            )
            
            # 保存为GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=prediction.dtype,
                crs=lr_metadata['crs'],
                transform=hr_transform,
                compress='lzw'
            ) as dst:
                dst.write(prediction, 1)
        else:
            # 保存为普通TIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=prediction.dtype,
                compress='lzw'
            ) as dst:
                dst.write(prediction, 1)
    
    def visualize_result(self, lr_water, hr_dem, hr_pred, output_path):
        """
        可视化预测结果
        
        Args:
            lr_water: 低分辨率水深
            hr_dem: 高分辨率DEM
            hr_pred: 预测的高分辨率水深
            output_path: 输出路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        scale_txt = f"{self.config.UPSCALE_FACTOR}x"
        
        # 低分辨率水深
        im1 = axes[0].imshow(lr_water, cmap='viridis')
        axes[0].set_title(f'Low-Resolution Water Depth', fontsize=14)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 高分辨率DEM
        im2 = axes[1].imshow(hr_dem, cmap='terrain')
        axes[1].set_title('High-Resolution DEM', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 预测的高分辨率水深
        im3 = axes[2].imshow(hr_pred, cmap='viridis')
        axes[2].set_title(f'Predicted High-Resolution Water Depth ({scale_txt})', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存可视化结果
        vis_path = output_path.replace('.tif', '_visualization.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"  可视化结果保存至: {vis_path}")
        plt.close()
    
    def predict_batch(self, lr_water_folder, hr_dem_folder, output_folder=None, visualize=False):
        """
        批量预测
        
        Args:
            lr_water_folder: 低分辨率水深文件夹
            hr_dem_folder: 高分辨率DEM文件夹
            output_folder: 输出文件夹（可选）
            visualize: 是否可视化结果
        """
        if output_folder is None:
            output_folder = self.config.OUTPUT_PRED_FOLDER
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有文件
        lr_files = sorted([
            os.path.join(lr_water_folder, f) 
            for f in os.listdir(lr_water_folder) 
            if f.endswith('.tif')
        ])
        
        print(f"\n找到 {len(lr_files)} 个文件需要处理")
        print("="*60)
        
        # 处理每个文件
        for i, lr_file in enumerate(lr_files, 1):
            print(f"\n[{i}/{len(lr_files)}] ", end="")
            
            # 获取对应的DEM文件
            dem_file = DataLoader.get_dem_path_from_water_path(lr_file, hr_dem_folder)
            
            # 输出路径
            basename = os.path.basename(lr_file).replace('.tif', '.tif')
            output_path = os.path.join(output_folder, basename)
            
            # 预测
            try:
                self.predict_single(lr_file, dem_file, output_path, visualize)
            except Exception as e:
                print(f"✗ 处理失败: {e}")
                continue
        
        print("\n" + "="*60)
        print(f"✓ 批量处理完成！结果保存在: {output_folder}")
    
    def evaluate(self, lr_water_path, hr_dem_path, hr_water_truth_path):
        """
        评估预测结果（需要真实标签）
        
        Args:
            lr_water_path: 低分辨率水深文件路径
            hr_dem_path: 高分辨率DEM文件路径
            hr_water_truth_path: 真实高分辨率水深文件路径
            
        Returns:
            评估指标字典
        """
        from losses import psnr_metric, ssim_metric
        
        # 预测
        hr_pred = self.predict_single(lr_water_path, hr_dem_path, visualize=False)
        
        # 加载真实标签
        with rasterio.open(hr_water_truth_path) as src:
            hr_truth = src.read(1).astype(np.float32)
        
        # 确保尺寸一致
        if hr_pred.shape != hr_truth.shape:
            print(f"Warning: Shape mismatch - pred: {hr_pred.shape}, truth: {hr_truth.shape}")
            return None
        
        # 转换为tensor
        hr_pred_tensor = tf.expand_dims(tf.expand_dims(hr_pred, 0), -1)
        hr_truth_tensor = tf.expand_dims(tf.expand_dims(hr_truth, 0), -1)
        
        # 计算指标
        psnr = psnr_metric(hr_truth_tensor, hr_pred_tensor).numpy()
        ssim = ssim_metric(hr_truth_tensor, hr_pred_tensor).numpy()
        mae = np.mean(np.abs(hr_truth - hr_pred))
        rmse = np.sqrt(np.mean((hr_truth - hr_pred) ** 2))
        
        metrics = {
            'PSNR': psnr,
            'SSIM': ssim,
            'MAE': mae,
            'RMSE': rmse
        }
        
        print("\n评估指标:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双流EDSR模型推理脚本')
    parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH,
                        help='模型路径')
    parser.add_argument('--mode', type=str, default='batch', choices=['single', 'batch'],
                        help='推理模式: single或batch')
    parser.add_argument('--lr_water', type=str, default=None,
                        help='低分辨率水深文件路径（single模式）')
    parser.add_argument('--hr_dem', type=str, default=None,
                        help='高分辨率DEM文件路径（single模式）')
    parser.add_argument('--lr_folder', type=str, default=Config.TEST_LR_WATER_FOLDER,
                        help='低分辨率水深文件夹路径（batch模式）')
    parser.add_argument('--dem_folder', type=str, default=Config.TEST_HR_DEM_FOLDER,
                        help='高分辨率DEM文件夹路径（batch模式）')
    parser.add_argument('--output', type=str, default=Config.OUTPUT_PRED_FOLDER,
                        help='输出路径')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    parser.add_argument('--evaluate', action='store_true',
                        help='是否评估结果（需要真实标签）')
    parser.add_argument('--hr_truth', type=str, default=None,
                        help='真实高分辨率水深文件路径（评估模式）')
    
    args = parser.parse_args()
    
    # 打印配置
    print("="*60)
    print("推理配置")
    print("="*60)
    print(f"模型路径: {args.model}")
    print(f"推理模式: {args.mode}")
    print(f"可视化: {'是' if args.visualize else '否'}")
    print("="*60)
    
    # 创建推理器
    inferencer = Inferencer(args.model, Config)
    
    # 执行推理
    if args.mode == 'single':
        if args.lr_water is None or args.hr_dem is None:
            print("错误: single模式需要指定 --lr_water 和 --hr_dem 参数")
            return
        
        if args.evaluate and args.hr_truth:
            # 评估模式
            inferencer.evaluate(args.lr_water, args.hr_dem, args.hr_truth)
        else:
            # 普通推理
            inferencer.predict_single(
                args.lr_water, 
                args.hr_dem, 
                args.output, 
                args.visualize
            )
    
    elif args.mode == 'batch':
        # 批量推理
        inferencer.predict_batch(
            args.lr_folder,
            args.dem_folder,
            args.output,
            args.visualize
        )
    
    print("\n✓ 推理完成！")


if __name__ == "__main__":
    main()


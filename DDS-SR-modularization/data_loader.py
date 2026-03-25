"""
Data loading and preprocessing module
"""

import os
import numpy as np
import tensorflow as tf
import rasterio
from scipy.ndimage import zoom

from config import Config


class DataLoader:
    """Data loader class"""
    
    def __init__(self, config=None):
        """
        Initialize data loader
        
        Args:
            config: Configuration object, defaults to Config
        """
        self.config = config or Config
        self.autotune = tf.data.AUTOTUNE
    
    @staticmethod
    def get_dem_path_from_water_path(water_path, dem_folder):
        """
        Get corresponding DEM path based on water depth file path
        """
        filename = os.path.basename(water_path)
        dem_path = os.path.join(dem_folder, filename)
        
        if not os.path.exists(dem_path):
            # Try other naming conventions
            base_name = filename.split('_')[0] if '_' in filename else filename.replace('.tif', '')
            dem_files = [f for f in os.listdir(dem_folder) if base_name in f and f.endswith('.tif')]
            if dem_files:
                dem_path = os.path.join(dem_folder, dem_files[0])
        
        return dem_path
    
    # Note: Need to change to instance method to access self.config
    def load_multimodal_data(self, lr_water_path, hr_water_path, hr_dem_path):
        """
        Load multimodal data: low-resolution water depth + high-resolution water depth (label) + high-resolution DEM
        """
        # Convert paths
        lr_water_str = lr_water_path.numpy().decode('utf-8')
        hr_water_str = hr_water_path.numpy().decode('utf-8')
        hr_dem_str = hr_dem_path.numpy().decode('utf-8')
        
        # Read low-resolution water depth
        with rasterio.open(lr_water_str) as src:
            lr_water = src.read(1).astype(np.float32)
        
        # Read high-resolution water depth (label)
        with rasterio.open(hr_water_str) as src:
            hr_water = src.read(1).astype(np.float32)
        
        # Read high-resolution DEM
        try:
            with rasterio.open(hr_dem_str) as src:
                hr_dem = src.read(1).astype(np.float32)
                # Normalize DEM to match water depth data range
                # Use DATA_RANGE_MAX from configuration
                hr_dem = (hr_dem - np.min(hr_dem)) / (np.max(hr_dem) - np.min(hr_dem) + 1e-8) * self.config.DATA_RANGE_MAX
        except Exception as e:
            print(f"Warning: Cannot read DEM file {hr_dem_str}, using zeros. Error: {e}")
            hr_dem = np.zeros_like(hr_water)
        
        # Ensure DEM size matches high-resolution water depth
        if hr_dem.shape != hr_water.shape:
            # print(f"Warning: DEM shape {hr_dem.shape} != water shape {hr_water.shape}, resizing...")
            zoom_factor = (hr_water.shape[0] / hr_dem.shape[0], hr_water.shape[1] / hr_dem.shape[1])
            hr_dem = zoom(hr_dem, zoom_factor, order=1)
        
        # Expand dimensions
        lr_water = np.expand_dims(lr_water, axis=-1)
        hr_water = np.expand_dims(hr_water, axis=-1)
        hr_dem = np.expand_dims(hr_dem, axis=-1)
        
        # Convert to tensor
        lr_water = tf.convert_to_tensor(lr_water, dtype=tf.float32)
        hr_water = tf.convert_to_tensor(hr_water, dtype=tf.float32)
        hr_dem = tf.convert_to_tensor(hr_dem, dtype=tf.float32)
        
        return lr_water, hr_dem, hr_water
    
    # Note: Need to change to instance method
    def load_inference_data(self, lr_water_path, hr_dem_path):
        """
        Load inference data (no label needed)
        """
        # Convert paths
        lr_water_str = lr_water_path.numpy().decode('utf-8') if hasattr(lr_water_path, 'numpy') else lr_water_path
        hr_dem_str = hr_dem_path.numpy().decode('utf-8') if hasattr(hr_dem_path, 'numpy') else hr_dem_path
        
        # 读取低分辨率水深并保存元数据
        with rasterio.open(lr_water_str) as src:
            lr_water = src.read(1).astype(np.float32)
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'width': src.width * self.config.UPSCALE_FACTOR,  # 使用配置中的上采样倍数
                'height': src.height * self.config.UPSCALE_FACTOR,
                'dtype': 'float32'
            }
        
        # 读取高分辨率DEM
        try:
            with rasterio.open(hr_dem_str) as src:
                hr_dem = src.read(1).astype(np.float32)
                # 归一化DEM
                hr_dem = (hr_dem - np.min(hr_dem)) / (np.max(hr_dem) - np.min(hr_dem) + 1e-8) * self.config.DATA_RANGE_MAX
        except Exception as e:
            print(f"Warning: Cannot read DEM file {hr_dem_str}, using zeros. Error: {e}")
            hr_dem = np.zeros((metadata['height'], metadata['width']), dtype=np.float32)
        
        # 扩展维度
        lr_water = np.expand_dims(lr_water, axis=-1)
        hr_dem = np.expand_dims(hr_dem, axis=-1)
        
        # 转换为tensor
        lr_water = tf.convert_to_tensor(lr_water, dtype=tf.float32)
        hr_dem = tf.convert_to_tensor(hr_dem, dtype=tf.float32)
        
        return lr_water, hr_dem, metadata
    
    def get_file_lists(self, mode='train'):
        """
        获取文件列表
        """
        if mode == 'train':
            hr_water_folder = self.config.TRAIN_HR_WATER_FOLDER
            lr_water_folder = self.config.TRAIN_LR_WATER_FOLDER
            dem_folder = self.config.TRAIN_HR_DEM_FOLDER
        elif mode == 'val':
            hr_water_folder = self.config.VAL_HR_WATER_FOLDER
            lr_water_folder = self.config.VAL_LR_WATER_FOLDER
            dem_folder = self.config.VAL_HR_DEM_FOLDER
        elif mode == 'test':
            hr_water_folder = None
            lr_water_folder = self.config.TEST_LR_WATER_FOLDER
            dem_folder = self.config.TEST_HR_DEM_FOLDER
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 获取低分辨率水深文件列表
        lr_water_files = sorted([
            os.path.join(lr_water_folder, f) 
            for f in os.listdir(lr_water_folder) 
            if f.endswith('.tif')
        ])
        
        # 获取高分辨率水深文件列表（测试模式不需要）
        if hr_water_folder:
            hr_water_files = sorted([
                os.path.join(hr_water_folder, f) 
                for f in os.listdir(hr_water_folder) 
                if f.endswith('.tif')
            ])
        else:
            hr_water_files = None
        
        # 获取DEM文件列表
        hr_dem_files = [
            self.get_dem_path_from_water_path(f, dem_folder) 
            for f in lr_water_files
        ]
        
        return lr_water_files, hr_water_files, hr_dem_files
    
    def create_dataset(self, mode='train'):
        """
        创建TensorFlow数据集
        """
        lr_water_files, hr_water_files, hr_dem_files = self.get_file_lists(mode)
        
        if mode in ['train', 'val']:
            # 训练和验证模式：需要标签
            dataset = tf.data.Dataset.from_tensor_slices((lr_water_files, hr_water_files, hr_dem_files))
            dataset = dataset.map(
                lambda lr, hr, dem: tf.py_function(
                    self.load_multimodal_data,  # 使用实例方法
                    [lr, hr, dem], 
                    [tf.float32, tf.float32, tf.float32]
                ),
                num_parallel_calls=self.autotune
            )
            # 重新组织数据格式：((lr_water, hr_dem), hr_water)
            dataset = dataset.map(lambda lr_water, hr_dem, hr_water: ((lr_water, hr_dem), hr_water))
        else:
            # 测试模式：不需要标签
            dataset = tf.data.Dataset.from_tensor_slices((lr_water_files, hr_dem_files))
            dataset = dataset.map(
                lambda lr, dem: tf.py_function(
                    self.load_inference_data,  # 使用实例方法
                    [lr, dem], 
                    [tf.float32, tf.float32, tf.string]
                ),
                num_parallel_calls=self.autotune
            )
        
        return dataset
    
    def prepare_dataset(self, dataset, mode='train', batch_size=None):
        """
        准备数据集用于训练/验证/测试
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE if mode in ['train', 'val'] else self.config.INFERENCE_BATCH_SIZE
        
        # 缓存数据
        if mode == 'train':
            dataset = dataset.take(self.config.CACHE_TRAIN_SIZE).cache()
        elif mode == 'val':
            dataset = dataset.take(self.config.CACHE_VAL_SIZE).cache()
        
        # 批处理
        dataset = dataset.batch(batch_size)
        
        # 训练模式重复数据
        if mode == 'train':
            dataset = dataset.repeat()
        
        # 预取
        dataset = dataset.prefetch(buffer_size=self.autotune)
        
        return dataset

"""
Trainer module
Manage model training process
"""

import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from config import Config
from model import build_dual_stream_model


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler: Linear Warmup + Cosine Annealing
    """
    
    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr=1e-7):
        """
        Initialize learning rate scheduler
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.decay_steps = total_steps - warmup_steps
    
    def __call__(self, step):
        """
        Calculate learning rate for current step
        """
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        
        # Warmup phase: Linear growth
        warmup_lr = self.initial_lr * step / warmup_steps
        
        # Cosine annealing phase
        decay_step = tf.maximum(step - warmup_steps, 0.0)
        cosine_decay = 0.5 * (1 + tf.cos(
            tf.constant(np.pi) * decay_step / self.decay_steps
        ))
        decay_lr = (self.initial_lr - self.min_lr) * cosine_decay + self.min_lr
        
        # Select learning rate based on current step
        return tf.cond(
            step < warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )
    
    def get_config(self):
        """Return configuration dictionary"""
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }
    
    # Operator overloading (support Keras internal operations)
    def __mul__(self, other):
        return self
    
    def __rmul__(self, other):
        return self
    
    def __truediv__(self, other):
        return self
    
    def __add__(self, other):
        return self


class LearningRateLogger(keras.callbacks.Callback):
    """Learning rate logging callback"""
    
    def on_epoch_end(self, epoch, logs=None):
        """Record learning rate at end of each epoch"""
        try:
            # Get optimizer
            if hasattr(self.model.optimizer, 'inner_optimizer'):
                optimizer = self.model.optimizer.inner_optimizer
            else:
                optimizer = self.model.optimizer
            
            # Get learning rate
            lr = optimizer.learning_rate
            
            # If scheduler, calculate current learning rate
            if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                current_step = optimizer.iterations
                lr_value = lr(current_step)
                if isinstance(lr_value, tf.Tensor):
                    lr_value = keras.backend.get_value(lr_value)
                else:
                    lr_value = float(lr_value)
            else:
                lr_value = keras.backend.get_value(lr)
            
            # Record to logs
            if logs is not None:
                logs['lr'] = float(lr_value)
            
            # Print for first 5 epochs and every 10 epochs
            if epoch < 5 or (epoch + 1) % 10 == 0:
                print(f"  → Learning Rate at epoch {epoch+1}: {lr_value:.7f}")
        except Exception as e:
            if epoch == 0:
                print(f"  Warning: Could not log learning rate: {e}")


class LossPrinterCallback(keras.callbacks.Callback):
    """Loss printing callback"""
    
    def on_epoch_end(self, epoch, logs=None):
        """Print detailed information at end of each epoch"""
        if logs is None:
            return
        
        print(f"\nEpoch {epoch+1:3d}: ", end="")
        
        # Print individual losses
        if 'recon_loss' in logs:
            print(f"Recon={logs['recon_loss']:.4f} ", end="")
        if 'grad_loss' in logs:
            print(f"Grad={logs['grad_loss']:.4f} ", end="")
        if 'smooth_loss' in logs:
            print(f"Smooth={logs['smooth_loss']:.6f} ", end="")
        if 'surface_loss' in logs:
            print(f"Surface={logs['surface_loss']:.4f} ", end="")
        if 'loss' in logs:
            print(f"Total={logs['loss']:.4f} ", end="")
        
        # Print evaluation metrics
        if 'PSNR' in logs:
            print(f"PSNR={logs['PSNR']:.2f} ", end="")
        if 'val_PSNR' in logs:
            print(f"Val_PSNR={logs['val_PSNR']:.2f} ", end="")
        if 'SSIM' in logs:
            print(f"SSIM={logs['SSIM']:.4f} ", end="")
        if 'val_SSIM' in logs:
            print(f"Val_SSIM={logs['val_SSIM']:.4f} ", end="")
        
        print()  # 换行


class Trainer:
    """训练器类"""
    
    def __init__(self, config=None):
        """
        初始化训练器
        
        Args:
            config: 配置对象
        """
        self.config = config or Config
        self.model = None
        self.history = None
        
        # 创建输出目录
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # 设置混合精度
        if self.config.USE_MIXED_PRECISION:
            self._setup_mixed_precision()
    
    def _setup_mixed_precision(self):
        """设置混合精度训练"""
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print('✓ Mixed Precision Enabled')
            print(f'  Compute dtype: {policy.compute_dtype}')
            print(f'  Variable dtype: {policy.variable_dtype}')
        except Exception as e:
            print(f'✗ Mixed Precision Failed: {e}')
            print('  Continuing with FP32...')
    
    def build_model(self):
        """构建模型"""
        print("\n构建双流EDSR模型...")
        self.model = build_dual_stream_model(
            num_filters=self.config.NUM_FILTERS,
            num_residual_blocks=self.config.NUM_RESIDUAL_BLOCKS,
            use_mixed_precision=self.config.USE_MIXED_PRECISION,
            data_max_value=self.config.DATA_RANGE_MAX,  # 传入最大值
            upscale_factor=self.config.UPSCALE_FACTOR,   # 传入上采样倍数
            ablation_mode=getattr(self.config, 'ABLATION_MODE', None) # 传入消融实验模式
        )
        print("模型构建完成！")
        return self.model
    
    def compile_model(self):
        """编译模型"""
        # 计算总步数
        total_steps = self.config.TOTAL_EPOCHS * self.config.STEPS_PER_EPOCH
        warmup_steps = self.config.WARMUP_EPOCHS * self.config.STEPS_PER_EPOCH
        
        # 创建学习率调度器
        lr_schedule = WarmupCosineDecay(
            initial_lr=self.config.INITIAL_LR,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.MIN_LR
        )
        
        print(f"\n学习率配置:")
        print(f"  初始学习率: {self.config.INITIAL_LR}")
        print(f"  最小学习率: {self.config.MIN_LR}")
        print(f"  Warmup步数: {warmup_steps} ({self.config.WARMUP_EPOCHS} epochs)")
        print(f"  总训练步数: {total_steps} ({self.config.TOTAL_EPOCHS} epochs)")
        
        # 创建优化器
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # 混合精度：包装优化器
        if self.config.USE_MIXED_PRECISION:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            print("  优化器: Adam + LossScaleOptimizer (Mixed Precision)")
        else:
            print("  优化器: Adam")
        
        # 编译模型
        self.model.compile(optimizer=optimizer, run_eagerly=False)
        print("模型编译完成！")
    
    def get_callbacks(self):
        """获取训练回调"""
        callbacks = []
        
        # 模型检查点
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.config.MODEL_SAVE_PATH,
            monitor=self.config.ES_MONITOR,
            mode=self.config.ES_MODE,
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # 早停
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=self.config.ES_MONITOR,
            mode=self.config.ES_MODE,
            patience=self.config.ES_PATIENCE,
            min_delta=self.config.ES_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 学习率记录
        callbacks.append(LearningRateLogger())
        
        # 损失打印
        callbacks.append(LossPrinterCallback())
        
        return callbacks
    
    def train(self, train_dataset, val_dataset):
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            
        Returns:
            训练历史
        """
        print("\n开始训练...")
        print("="*85)
        print("Epoch | Recon_L | Grad_L | Smooth_L | Surf_L | Total_L | PSNR | Val_PSNR | SSIM | Val_SSIM")
        print("="*85)
        
        start_time = time.time()
        
        # 训练
        self.history = self.model.fit(
            train_dataset,
            epochs=self.config.TOTAL_EPOCHS,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            validation_data=val_dataset,
            callbacks=self.get_callbacks(),
            verbose=self.config.VERBOSE
        )
        
        # 计算训练时间
        training_time = time.time() - start_time
        print(f"\n训练完成！用时: {training_time:.2f}秒 ({training_time/3600:.2f}小时)")
        
        return self.history
    
    def save_model(self):
        """保存模型"""
        self.model.save(self.config.FINAL_MODEL_PATH)
        print(f"最终模型已保存: {self.config.FINAL_MODEL_PATH}")
    
    def save_history(self):
        """保存训练历史"""
        # 清理不可序列化的对象
        history_dict = dict(self.history.history)
        if 'learning_rate' in history_dict:
            lr_values = history_dict['learning_rate']
            if lr_values and not isinstance(lr_values[0], (int, float)):
                del history_dict['learning_rate']
        
        # 保存
        with open(self.config.HISTORY_SAVE_PATH, 'wb') as f:
            pickle.dump(history_dict, f)
        print(f"训练历史已保存: {self.config.HISTORY_SAVE_PATH}")
    
    def load_model(self, model_path=None):
        """
        加载模型
        
        Args:
            model_path: 模型路径，默认使用配置中的最佳模型路径
        """
        from model import DualStreamEDSR
        
        if model_path is None:
            model_path = self.config.MODEL_SAVE_PATH
        
        # 提供自定义对象以正确加载模型
        custom_objects = {
            'DualStreamEDSR': DualStreamEDSR
        }
        
        self.model = keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects=custom_objects
        )
        print(f"模型已加载: {model_path}")
        return self.model

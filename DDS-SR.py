# -*-coding:utf-8 -*-
"""
"""
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
import rasterio
import pickle
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# Set tf.data auto-tuning parameters
AUTOTUNE = tf.data.AUTOTUNE

# ============ Training configuration parameters (adjustable as needed) ============
# Learning rate configuration
INITIAL_LR = 1e-4          # Initial learning rate
MIN_LR = 1e-6              # Minimum learning rate
WARMUP_EPOCHS = 5          # Warmup epochs
TOTAL_EPOCHS = 200        # Total training epochs
STEPS_PER_EPOCH = 100      # Steps per epoch

# Batch size
BATCH_SIZE = 4             # Adjust based on GPU memory [4/8/16]

# Mixed precision training
USE_MIXED_PRECISION = True  # Whether to enable mixed precision (requires supported GPU)

# Early Stopping configuration
ES_PATIENCE = 50          # Patience value
ES_MIN_DELTA = 0.1         # Minimum PSNR improvement
# ============ End of configuration parameters ============

# Check available GPU count
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Enable mixed precision training
if USE_MIXED_PRECISION:
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print('✓ Mixed Precision Enabled')
        print('  Compute dtype: %s' % policy.compute_dtype)
        print('  Variable dtype: %s' % policy.variable_dtype)
    except Exception as e:
        print(f'✗ Mixed Precision Failed: {e}')
        print('  Continuing with FP32...')
else:
    print('Mixed Precision Disabled (using FP32)')

#### Dataset path settings
# Training set paths
train_hr_water_folder = "date/Label_1m_train/"  # High-resolution water depth (1m)
train_lr_water_folder = "date/Feature_4m_train/"  # Low-resolution water depth (4m)
train_hr_dem_folder = "date/DEM-01m/"  # High-resolution DEM (1m)

# Validation set paths
val_hr_water_folder = "date/Label_1m_val/"
val_lr_water_folder = "date/Feature_4m_val/"
val_hr_dem_folder = "date/DEM-01m/"

# Super-resolution scale factor
scale_factors = 4 # e.g., to upscale from 1m to 4m, enter 4
# Save paths
final_model_save_path = 'out_save/DDS-SR_4x_final.h5'
best_model_save_path = 'out_save/DDS_SR_final_4x.h5'
history_save_path = 'out_save/DDS-SR-4x.pkl'

# Get file lists
train_hr_water_files = sorted([os.path.join(train_hr_water_folder, f) for f in os.listdir(train_hr_water_folder) if f.endswith('.tif')])
train_lr_water_files = sorted([os.path.join(train_lr_water_folder, f) for f in os.listdir(train_lr_water_folder) if f.endswith('.tif')])

# DEM files need to match water depth file names
def get_dem_path_from_water_path(water_path, dem_folder):
    """Get DEM path based on water depth file path"""
    filename = os.path.basename(water_path)
    # Assume DEM file naming matches water depth file, or adjust based on actual situation
    dem_path = os.path.join(dem_folder, filename)
    if not os.path.exists(dem_path):
        # If corresponding DEM not found, try other naming rules
        # This needs to be adjusted based on actual data structure
        base_name = filename.split('_')[0] if '_' in filename else filename.replace('.tif', '')
        dem_files = [f for f in os.listdir(dem_folder) if base_name in f and f.endswith('.tif')]
        if dem_files:
            dem_path = os.path.join(dem_folder, dem_files[0])
    return dem_path

# Create DEM file list
train_hr_dem_files = [get_dem_path_from_water_path(f, train_hr_dem_folder) for f in train_hr_water_files]
val_hr_water_files = sorted([os.path.join(val_hr_water_folder, f) for f in os.listdir(val_hr_water_folder) if f.endswith('.tif')])
val_lr_water_files = sorted([os.path.join(val_lr_water_folder, f) for f in os.listdir(val_lr_water_folder) if f.endswith('.tif')])
val_hr_dem_files = [get_dem_path_from_water_path(f, val_hr_dem_folder) for f in val_hr_water_files]


def load_multimodal_data(lr_water_path, hr_water_path, hr_dem_path):
    """
    Load multimodal data: Low-resolution water depth + High-resolution water depth (label) + High-resolution DEM
    Returns three independent tensors: lr_water, hr_dem, hr_water
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
            # Normalize DEM to reasonable range
            hr_dem = (hr_dem - np.min(hr_dem)) / (np.max(hr_dem) - np.min(hr_dem) + 1e-8) * 255.0
    except Exception as e:
        print(f"Warning: Cannot read DEM file {hr_dem_str}, using zeros. Error: {e}")
        hr_dem = np.zeros_like(hr_water)
    
    # Ensure DEM size matches high-resolution water depth
    if hr_dem.shape != hr_water.shape:
        print(f"Warning: DEM shape {hr_dem.shape} != water shape {hr_water.shape}, resizing...")
        from scipy.ndimage import zoom
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
    
    # Return three independent tensors: lr_water, hr_dem, hr_water
    return lr_water, hr_dem, hr_water


# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_lr_water_files, train_hr_water_files, train_hr_dem_files))
train_dataset = train_dataset.map(
    lambda lr, hr, dem: tf.py_function(load_multimodal_data, [lr, hr, dem], 
                                       [tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=AUTOTUNE
)
# Reorganize data format: ((lr_water, hr_dem), hr_water)
train_dataset = train_dataset.map(lambda lr_water, hr_dem, hr_water: ((lr_water, hr_dem), hr_water))

val_dataset = tf.data.Dataset.from_tensor_slices((val_lr_water_files, val_hr_water_files, val_hr_dem_files))
val_dataset = val_dataset.map(
    lambda lr, hr, dem: tf.py_function(load_multimodal_data, [lr, hr, dem], 
                                       [tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=AUTOTUNE
)
# Reorganize data format: ((lr_water, hr_dem), hr_water)
val_dataset = val_dataset.map(lambda lr_water, hr_dem, hr_water: ((lr_water, hr_dem), hr_water))

# Cache data
train_cache = train_dataset.take(700).cache()
val_cache = val_dataset.take(300).cache()

print("Number of training samples:", len(train_hr_water_files))
print("Number of validation samples:", len(val_hr_water_files))

# Check data shape
example_sample = next(iter(train_cache))
print("LR water shape:", example_sample[0][0].shape)
print("HR DEM shape:", example_sample[0][1].shape)
print("HR water (label) shape:", example_sample[1].shape)


#### Build dataset object
def dataset_object(dataset_cache, training=True, batch_size=8):
    ds = dataset_cache
    ds = ds.batch(batch_size)
    if training:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = dataset_object(train_cache, training=True, batch_size=BATCH_SIZE)
val_ds = dataset_object(val_cache, training=False, batch_size=BATCH_SIZE)


#### Define evaluation metrics
def PSNR(high_resolution, super_resolution):
    """Peak Signal-to-Noise Ratio"""
    return tf.image.psnr(high_resolution, super_resolution, max_val=255)


def SSIM(high_resolution, super_resolution, max_val=255):
    """Structural Similarity Index"""
    high_resolution = tf.image.convert_image_dtype(high_resolution, tf.float32)
    super_resolution = tf.image.convert_image_dtype(super_resolution, tf.float32)
    ssim_value = tf.image.ssim(high_resolution, super_resolution, max_val=max_val)
    return ssim_value


#### Physics-constrained loss function
def physics_informed_loss(y_true, y_pred, dem_hr, x_lr):
    """
    Physics-constrained loss = Reconstruction loss + Physics constraints + Volume consistency
    Returns: (Total loss, Dictionary of individual losses)
    """
    # Mixed precision training: Ensure all tensor types are consistent
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dem_hr = tf.cast(dem_hr, tf.float32)
    x_lr = tf.cast(x_lr, tf.float32)
    
    # 1. Basic reconstruction loss (MAE)
    reconstruction_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # 2. Gradient consistency constraint
    # Terrain gradient
    dem_grad_x = dem_hr[:, :, 1:, :] - dem_hr[:, :, :-1, :]
    dem_grad_y = dem_hr[:, 1:, :, :] - dem_hr[:, :-1, :, :]

    # Water depth gradient
    water_grad_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    water_grad_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]

    # Gradient correlation: Higher terrain should have shallower water (negative correlation)
    # Use ReLU to only penalize positive correlation
    gradient_consistency_x = tf.reduce_mean(tf.nn.relu(dem_grad_x * water_grad_x))
    gradient_consistency_y = tf.reduce_mean(tf.nn.relu(dem_grad_y * water_grad_y))
    gradient_consistency = gradient_consistency_x + gradient_consistency_y

    # 3. Local smoothness constraint (avoid over-penalizing real boundaries)
    smoothness_x = tf.reduce_mean(tf.square(water_grad_x))
    smoothness_y = tf.reduce_mean(tf.square(water_grad_y))
    smoothness_loss = smoothness_x + smoothness_y

    # 4. Water surface continuity constraint (Water surface elevation = DEM + Water depth should be continuous)
    water_surface = dem_hr + y_pred
    surface_grad_x = water_surface[:, :, 1:, :] - water_surface[:, :, :-1, :]
    surface_grad_y = water_surface[:, 1:, :, :] - water_surface[:, :-1, :, :]
    surface_smoothness = tf.reduce_mean(tf.square(surface_grad_x)) + tf.reduce_mean(tf.square(surface_grad_y))

    # 5. Volume consistency constraint: Downsample 1m results back to 4m, align with low-resolution input
    # Use average pooling to simulate the aggregation process from high to low resolution, ensuring water volume conservation
    # 4x super-resolution, so ksize=4, strides=4
    y_pred_downsampled = tf.nn.avg_pool2d(y_pred, ksize=4, strides=4, padding='VALID')
    # Note: x_lr is also 4m resolution, can be compared directly
    volume_consistency = tf.reduce_mean(tf.abs(y_pred_downsampled - x_lr))

    # Combined loss (weights adjustable)
    total_loss = reconstruction_loss + \
                 0.05 * gradient_consistency + \
                 0.001 * smoothness_loss + \
                 0.01 * surface_smoothness + \
                 0.1 * volume_consistency  # Volume consistency weight, usually given larger weight as it's both a physics constraint and data constraint

    # Return total loss and dictionary of individual losses
    loss_dict = {
        'recon': reconstruction_loss,
        'grad': gradient_consistency,
        'smooth': smoothness_loss,
        'surface': surface_smoothness,
        'volume': volume_consistency,
        'total': total_loss
    }

    return total_loss, loss_dict


#### Define network components
def ResBlock(inputs):
    """Residual block"""
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


def ChannelAttention(inputs, reduction_ratio=16):
    """Channel attention module"""
    squeeze = layers.GlobalAveragePooling2D()(inputs)
    squeeze = layers.Dense(inputs.shape[-1] // reduction_ratio, activation='relu')(squeeze)
    excitation = layers.Dense(inputs.shape[-1], activation='sigmoid')(squeeze)
    excitation = layers.Reshape((1, 1, inputs.shape[-1]))(excitation)
    scale = layers.Multiply()([inputs, excitation])
    return scale


def CrossModalAttention(water_features, dem_features):
    """
    Cross-modal attention: Use DEM features to guide water depth features
    """
    # DEM generates spatial attention map
    spatial_attention = layers.Conv2D(64, 3, padding="same", activation='relu')(dem_features)
    spatial_attention = layers.Conv2D(1, 1, activation='sigmoid')(spatial_attention)
    
    # Apply attention to water depth features
    water_attended = layers.Multiply()([water_features, spatial_attention])
    
    # Feature fusion
    concat = layers.Concatenate()([water_attended, dem_features])
    fused = layers.Conv2D(64, 3, padding="same", activation='relu')(concat)
    fused = layers.Conv2D(64, 3, padding="same")(fused)
    
    return fused


def Upsampling(inputs, factor=2, **kwargs):
    """Upsampling block"""
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


#### Custom model class
class DualStreamEDSR(tf.keras.Model):
    """Dual-stream EDSR model: Integrated physics-constrained loss"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Basic metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="PSNR")
        self.ssim_metric = keras.metrics.Mean(name="SSIM")

        # Individual loss trackers
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")
        self.smooth_loss_tracker = keras.metrics.Mean(name="smooth_loss")
        self.surface_loss_tracker = keras.metrics.Mean(name="surface_loss")
        self.volume_loss_tracker = keras.metrics.Mean(name="volume_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker, self.psnr_metric, self.ssim_metric,
            self.recon_loss_tracker, self.grad_loss_tracker,
            self.smooth_loss_tracker, self.surface_loss_tracker,
            self.volume_loss_tracker
        ]

    def train_step(self, data):
        (x_water, x_dem), y = data

        with tf.GradientTape() as tape:
            y_pred = self([x_water, x_dem], training=True)
            # Use physics-constrained loss, returns total loss and individual losses
            # Note: Pass x_water for volume loss calculation
            loss, loss_dict = physics_informed_loss(y, y_pred, x_dem, x_water)
            
            # Mixed precision training: Scale loss inside GradientTape (critical!)
            if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        
        # Mixed precision training: Unscale gradients
        if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(PSNR(y, y_pred))
        self.ssim_metric.update_state(SSIM(y, y_pred))

        # Update individual loss metrics
        self.recon_loss_tracker.update_state(loss_dict['recon'])
        self.grad_loss_tracker.update_state(loss_dict['grad'])
        self.smooth_loss_tracker.update_state(loss_dict['smooth'])
        self.surface_loss_tracker.update_state(loss_dict['surface'])
        self.volume_loss_tracker.update_state(loss_dict['volume'])

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (x_water, x_dem), y = data
        y_pred = self([x_water, x_dem], training=False)
        # Note: Pass x_water for volume loss calculation
        loss, loss_dict = physics_informed_loss(y, y_pred, x_dem, x_water)

        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(PSNR(y, y_pred))
        self.ssim_metric.update_state(SSIM(y, y_pred))

        # Update individual loss metrics
        self.recon_loss_tracker.update_state(loss_dict['recon'])
        self.grad_loss_tracker.update_state(loss_dict['grad'])
        self.smooth_loss_tracker.update_state(loss_dict['smooth'])
        self.surface_loss_tracker.update_state(loss_dict['surface'])
        self.volume_loss_tracker.update_state(loss_dict['volume'])

        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        x_water, x_dem = data
        x_water = tf.cast(x_water, tf.float32)
        x_dem = tf.cast(x_dem, tf.float32)
        super_resolution_img = self([x_water, x_dem], training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        return super_resolution_img


#### Build dual-stream model
def make_dual_stream_model(num_filters=64, num_of_residual_blocks=16):
    """
    Dual-stream architecture: Water depth stream + DEM stream
    """
    # ===== Input layer =====
    water_input = layers.Input(shape=(None, None, 1), name='water_depth_4m')
    dem_input = layers.Input(shape=(None, None, 1), name='dem_1m')
    
    # ===== Water depth stream (low resolution) =====
    water_x = layers.Rescaling(scale=1.0 / 255)(water_input)
    water_x = layers.Conv2D(num_filters, 3, padding="same")(water_x)
    water_features = water_x
    
    # Water depth residual blocks
    for _ in range(num_of_residual_blocks // 2):
        water_features = ResBlock(water_features)
    
    water_features = layers.Conv2D(num_filters, 3, padding="same")(water_features)
    water_features = layers.Add()([water_x, water_features])
    water_features = ChannelAttention(water_features)
    
    # ===== DEM stream (high resolution) =====
    dem_x = layers.Rescaling(scale=1.0 / 255)(dem_input)
    dem_x = layers.Conv2D(num_filters, 3, padding="same")(dem_x)
    dem_features = dem_x
    
    # DEM residual blocks
    for _ in range(num_of_residual_blocks // 2):
        dem_features = ResBlock(dem_features)
    
    dem_features = layers.Conv2D(num_filters, 3, padding="same")(dem_features)
    dem_features = layers.Add()([dem_x, dem_features])
    dem_features = ChannelAttention(dem_features)
    
    # ===== Upsample water depth features to 1m resolution =====

    water_upsampled = Upsampling(water_features, factor=2)
    for i in range(int(math.log2(scale_factors)-1)):
        water_upsampled = Upsampling(water_upsampled, factor=2)
    
    # ===== Cross-modal attention fusion =====
    fused_features = CrossModalAttention(water_upsampled, dem_features)
    
    # ===== Residual refinement =====
    for _ in range(num_of_residual_blocks // 2):
        fused_features = ResBlock(fused_features)
    
    # ===== Output layer =====
    output = layers.Conv2D(1, 3, padding="same")(fused_features)
    output = layers.Rescaling(scale=255)(output)
    
    # Mixed precision training: Force output to float32 (avoid type mismatch in loss calculation)
    if USE_MIXED_PRECISION:
        output = layers.Activation('linear', dtype='float32')(output)
    
    return DualStreamEDSR([water_input, dem_input], output)


# Create model
print("\nCreating dual-stream model...")
model = make_dual_stream_model(num_filters=64, num_of_residual_blocks=16)
model.summary()

#### Training configuration
# Custom learning rate scheduler: Warmup + Cosine annealing
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler: Linear Warmup + Cosine annealing
    """
    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.decay_steps = total_steps - warmup_steps
    
    def __call__(self, step):
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
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }
    
    # 添加运算符重载以支持 Keras 的内部操作
    def __mul__(self, other):
        """支持乘法运算（Keras 历史记录需要）"""
        return self
    
    def __rmul__(self, other):
        """支持反向乘法运算"""
        return self
    
    def __truediv__(self, other):
        """支持除法运算"""
        return self
    
    def __add__(self, other):
        """支持加法运算"""
        return self

# 计算总步数
total_steps = TOTAL_EPOCHS * STEPS_PER_EPOCH
warmup_steps = WARMUP_EPOCHS * STEPS_PER_EPOCH

# 创建学习率调度器
lr_schedule = WarmupCosineDecay(
    initial_lr=INITIAL_LR,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    min_lr=MIN_LR
)

print(f"\n学习率配置:")
print(f"  初始学习率: {INITIAL_LR}")
print(f"  最小学习率: {MIN_LR}")
print(f"  Warmup步数: {warmup_steps} ({WARMUP_EPOCHS} epochs)")
print(f"  总训练步数: {total_steps} ({TOTAL_EPOCHS} epochs)")

# 创建优化器
optim = keras.optimizers.Adam(learning_rate=lr_schedule)

# 如果使用混合精度，包装优化器
if USE_MIXED_PRECISION:
    optim = mixed_precision.LossScaleOptimizer(optim)
    print("  优化器: Adam + LossScaleOptimizer (Mixed Precision)")
else:
    print("  优化器: Adam")

# 编译模型（不记录学习率到历史以避免序列化问题）
model.compile(optimizer=optim, run_eagerly=False)

# 回调函数
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath= best_model_save_path,
    monitor='val_PSNR',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_PSNR',           # 监控PSNR而非loss
    mode='max',                   # PSNR越大越好
    patience=ES_PATIENCE,         # 使用配置的耐心值
    min_delta=ES_MIN_DELTA,       # 最小提升量
    restore_best_weights=True,
    verbose=1
)


# 学习率记录回调
class LearningRateLogger(keras.callbacks.Callback):
    """记录每个epoch的学习率"""
    def on_epoch_end(self, epoch, logs=None):
        try:
            # 获取优化器
            if hasattr(self.model.optimizer, 'inner_optimizer'):
                # 混合精度情况
                optimizer = self.model.optimizer.inner_optimizer
            else:
                optimizer = self.model.optimizer
            
            # 获取学习率
            lr = optimizer.learning_rate
            
            # 如果是调度器，计算当前学习率
            if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                # 获取当前步数
                current_step = optimizer.iterations
                lr_value = lr(current_step)
                if isinstance(lr_value, tf.Tensor):
                    lr_value = keras.backend.get_value(lr_value)
                else:
                    lr_value = float(lr_value)
            else:
                lr_value = keras.backend.get_value(lr)
            
            # 记录到logs中（使用数值而不是调度器对象）
            if logs is not None:
                logs['lr'] = float(lr_value)
            
            # 前5轮和每10轮打印一次
            if epoch < 5 or (epoch + 1) % 10 == 0:
                print(f"  → Learning Rate at epoch {epoch+1}: {lr_value:.7f}")
        except Exception as e:
            # 如果获取学习率失败，不影响训练
            if epoch == 0:
                print(f"  Warning: Could not log learning rate: {e}")

# 自定义回调函数：在每轮结束后打印各项损失
class LossPrinterCallback(tf.keras.callbacks.Callback):
    """自定义回调：每轮结束后打印各项损失详情"""

    def on_epoch_end(self, epoch, logs=None):
        """每轮结束时打印详细信息"""
        if logs is None:
            return
        
        print(f"\nEpoch {epoch+1:3d}: ", end="")

        # 打印各项损失
        if 'recon_loss' in logs:
            print(f"{logs['recon_loss']:.4f} ", end="")
        if 'grad_loss' in logs:
            print(f"{logs['grad_loss']:.4f} ", end="")
        if 'smooth_loss' in logs:
            print(f"{logs['smooth_loss']:.6f} ", end="")
        if 'surface_loss' in logs:
            print(f"{logs['surface_loss']:.4f} ", end="")
        if 'volume_loss' in logs:
            print(f"{logs['volume_loss']:.4f} ", end="")
        if 'loss' in logs:
            print(f"{logs['loss']:.4f} ", end="")

        # 打印评估指标
        if 'PSNR' in logs:
            print(f"{logs['PSNR']:.2f} ", end="")
        if 'val_PSNR' in logs:
            print(f"{logs['val_PSNR']:.2f} ", end="")
        if 'SSIM' in logs:
            print(f"{logs['SSIM']:.4f} ", end="")
        if 'val_SSIM' in logs:
            print(f"{logs['val_SSIM']:.4f} ", end="")

        print()  # 换行

# 记录训练开始时间
start_time = time.time()

# 开始训练
print("\n开始训练双流模型...")
print("Epoch | Recon_L | Grad_L | Smooth_L | Surf_L | Vol_L  | Total_L | PSNR | Val_PSNR | SSIM | Val_SSIM")
print("-" * 95)

history = model.fit(
    train_ds,
    epochs=TOTAL_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    callbacks=[
        checkpoint_callback, 
        early_stopping, 
        # reduce_lr,  # 已移除：与余弦退火学习率调度器冲突
        LearningRateLogger(),
        LossPrinterCallback()
    ]
)

# 保存训练历史（清理学习率调度器对象）
os.makedirs(os.path.dirname(history_save_path), exist_ok=True)

# 清理历史记录中的不可序列化对象
history_dict = dict(history.history)
# 移除可能导致序列化问题的键
if 'learning_rate' in history_dict:
    # 如果learning_rate是调度器对象，转换为数值列表
    lr_values = history_dict['learning_rate']
    if not isinstance(lr_values[0] if lr_values else None, (int, float)):
        del history_dict['learning_rate']

with open(history_save_path, 'wb') as file_pi:
    pickle.dump(history_dict, file_pi)

# 计算训练总时间
end_time = time.time()
training_time = end_time - start_time

# 保存最终模型

model.save(final_model_save_path)

print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

#### 可视化训练过程
# 创建2x2的子图布局
fig = plt.figure(figsize=(16, 12))

# 1. 总损失曲线
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='train total loss', linewidth=2, color='red')
plt.plot(history.history['val_loss'], label='val total loss', linewidth=2, linestyle='--', color='red')
plt.xlabel('Epochs', size=12)
plt.ylabel('Total Loss', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Total Loss (4x SR)")

# 2. 各项损失曲线
plt.subplot(2, 2, 2)
plt.plot(history.history.get('recon_loss', []), label='recon loss', linewidth=2, color='blue')
plt.plot(history.history.get('grad_loss', []), label='grad loss', linewidth=2, color='green')
plt.plot(history.history.get('smooth_loss', []), label='smooth loss', linewidth=2, color='orange')
plt.plot(history.history.get('surface_loss', []), label='surface loss', linewidth=2, color='purple')
plt.plot(history.history.get('volume_loss', []), label='volume loss', linewidth=2, color='brown')
plt.xlabel('Epochs', size=12)
plt.ylabel('Component Loss', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Component Losses (4x SR)")
plt.yscale('log')  # 对数尺度，更好显示小数值

# 3. PSNR曲线
plt.subplot(2, 2, 3)
plt.plot(history.history['PSNR'], label='train PSNR', linewidth=2, color='darkgreen')
plt.plot(history.history['val_PSNR'], label='val PSNR', linewidth=2, linestyle='--', color='darkgreen')
plt.xlabel('Epochs', size=12)
plt.ylabel('PSNR (dB)', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("PSNR (4x SR)")

# 4. SSIM曲线
plt.subplot(2, 2, 4)
plt.plot(history.history['SSIM'], label='train SSIM', linewidth=2, color='darkblue')
plt.plot(history.history['val_SSIM'], label='val SSIM', linewidth=2, linestyle='--', color='darkblue')
plt.xlabel('Epochs', size=12)
plt.ylabel('SSIM', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("SSIM (4x SR)")

plt.tight_layout()
plt.savefig('out_save/dual_stream_training_curves3.png', dpi=300, bbox_inches='tight')
plt.show()

# 额外显示各项损失的数值统计
print("\n" + "="*60)
print("Loss Component Statistics (Final Epoch)")
print("="*60)
if 'recon_loss' in history.history:
    print(".4f")
if 'grad_loss' in history.history:
    print(".4f")
if 'smooth_loss' in history.history:
    print(".6f")
if 'surface_loss' in history.history:
    print(".4f")
if 'volume_loss' in history.history:
    print(".4f")
print(".4f")
print("="*60)

print("\n训练完成！模型和历史记录已保存。")


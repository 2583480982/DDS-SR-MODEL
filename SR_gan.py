# -*-coding:utf-8 -*-
"""
Author: SRGAN comparison model - DEM-guided water depth super-resolution
Date: December 6, 2025
Architecture: SRGAN (Generator + Discriminator + Adversarial loss)
Input: Low-resolution water depth (4m) + High-resolution DEM (1m)
Output: High-resolution water depth (1m)
Reference: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (CVPR 2017)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
import rasterio
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision  # Import mixed precision

# Set tf.data auto-tuning parameters
AUTOTUNE = tf.data.AUTOTUNE

# Enable mixed precision training (FP16)
# This will significantly reduce memory usage and accelerate training
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print('✓ Mixed Precision Enabled')
except Exception as e:
    print(f'Warning: Mixed Precision failed: {e}')

# ============ Training configuration parameters ============
BATCH_SIZE = 4  # Reduce Batch Size to solve OOM
TOTAL_EPOCHS = 200
STEPS_PER_EPOCH = 100
INITIAL_LR_G = 1e-4
INITIAL_LR_D = 1e-4
ES_PATIENCE = 50
ES_MIN_DELTA = 0.1

# Loss weights
ADVERSARIAL_WEIGHT = 1e-3  # Adversarial loss weight
CONTENT_WEIGHT = 1.0       # Content loss weight
# ============ End of configuration parameters ============

# Check available GPU count
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#### Dataset path settings
train_hr_water_folder = "date/Label_1m_train/"
train_lr_water_folder = "date/Feature_4m_train/"
train_hr_dem_folder = "date/DEM-01m/"

val_hr_water_folder = "date/Label_1m_val/"
val_lr_water_folder = "date/Feature_4m_val/"
val_hr_dem_folder = "date/DEM-01m/"

# Get file lists
train_hr_water_files = sorted([os.path.join(train_hr_water_folder, f) for f in os.listdir(train_hr_water_folder) if f.endswith('.tif')])
train_lr_water_files = sorted([os.path.join(train_lr_water_folder, f) for f in os.listdir(train_lr_water_folder) if f.endswith('.tif')])

# DEM file matching function
def get_dem_path_from_water_path(water_path, dem_folder):
    """Get DEM path based on water depth file path"""
    filename = os.path.basename(water_path)
    dem_path = os.path.join(dem_folder, filename)
    if not os.path.exists(dem_path):
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
    """Load multimodal data: Low-resolution water depth + High-resolution water depth (label) + High-resolution DEM"""
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
    
    return lr_water, hr_dem, hr_water


# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_lr_water_files, train_hr_water_files, train_hr_dem_files))
train_dataset = train_dataset.map(
    lambda lr, hr, dem: tf.py_function(load_multimodal_data, [lr, hr, dem], 
                                       [tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=AUTOTUNE
)
train_dataset = train_dataset.map(lambda lr_water, hr_dem, hr_water: ((lr_water, hr_dem), hr_water))

val_dataset = tf.data.Dataset.from_tensor_slices((val_lr_water_files, val_hr_water_files, val_hr_dem_files))
val_dataset = val_dataset.map(
    lambda lr, hr, dem: tf.py_function(load_multimodal_data, [lr, hr, dem], 
                                       [tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=AUTOTUNE
)
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


#### Residual block (for generator)
def residual_block(x, filters=64):
    """Residual block: Conv -> BN -> PReLU -> Conv -> BN + Skip"""
    skip = x
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([skip, x])
    return x


def upsample_block(x, filters=64):
    """Upsampling block: Conv -> PixelShuffle -> PReLU"""
    x = layers.Conv2D(filters * 4, 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, block_size=2)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    return x


#### Build generator (Generator)
def make_generator(num_res_blocks=16, num_filters=64):
    """
    SRGAN generator:
    Input: Low-resolution water depth (4m) + High-resolution DEM (1m)
    Output: High-resolution water depth (1m)
    """
    # ===== Input layer =====
    water_input = layers.Input(shape=(None, None, 1), name='water_depth_4m')
    dem_input = layers.Input(shape=(None, None, 1), name='dem_1m')
    
    # ===== Preprocessing =====
    water_x = layers.Rescaling(scale=1.0 / 255)(water_input)
    dem_x = layers.Rescaling(scale=1.0 / 255)(dem_input)
    
    # Upsample low-resolution water depth to 1m (4x)
    water_upsampled = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(water_x)
    
    # Concatenate water depth and DEM
    x = layers.Concatenate()([water_upsampled, dem_x])
    
    # ===== Initial feature extraction =====
    x = layers.Conv2D(num_filters, 9, padding='same')(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    skip_main = x
    
    # ===== Residual block stacking =====
    for _ in range(num_res_blocks):
        x = residual_block(x, num_filters)
    
    # ===== Skip connection =====
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([skip_main, x])
    
    # ===== Upsampling layer (already upsampled via bilinear interpolation, only feature enhancement here) =====
    # x = upsample_block(x, num_filters)  # 2x
    # x = upsample_block(x, num_filters)  # 4x (total)
    
    # ===== Output layer =====
    x = layers.Conv2D(1, 9, padding='same')(x)
    x = layers.Rescaling(scale=255)(x)
    # Mixed precision: ensure output layer is float32
    output = layers.Activation('linear', dtype='float32')(x)
    
    model = keras.Model([water_input, dem_input], output, name='Generator')
    return model


#### Build discriminator (Discriminator)
def make_discriminator(num_filters=64):
    """
    SRGAN discriminator: Determine if input image is real high-resolution or generated
    Input: High-resolution water depth (1m) + High-resolution DEM (1m)
    Output: Real probability
    """
    # ===== Input layer =====
    water_input = layers.Input(shape=(None, None, 1), name='water_hr')
    dem_input = layers.Input(shape=(None, None, 1), name='dem_hr')
    
    # ===== Preprocessing =====
    water_x = layers.Rescaling(scale=1.0 / 255)(water_input)
    dem_x = layers.Rescaling(scale=1.0 / 255)(dem_input)
    
    # Concatenate water depth and DEM
    x = layers.Concatenate()([water_x, dem_x])
    
    # ===== Convolutional layer stacking =====
    # Block 1
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Block 2
    x = layers.Conv2D(num_filters, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Block 3
    x = layers.Conv2D(num_filters * 2, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Block 4
    x = layers.Conv2D(num_filters * 2, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Block 5
    x = layers.Conv2D(num_filters * 4, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Block 6
    x = layers.Conv2D(num_filters * 4, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Block 7
    x = layers.Conv2D(num_filters * 8, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Block 8
    x = layers.Conv2D(num_filters * 8, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # ===== Global pooling + fully connected =====
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1)(x) # Remove sigmoid activation, use from_logits=True in loss for more stability
    # Mixed precision: ensure output layer is float32
    output = layers.Activation('sigmoid', dtype='float32')(x)
    
    model = keras.Model([water_input, dem_input], output, name='Discriminator')
    return model


#### Custom SRGAN model class
class SRGAN(keras.Model):
    """SRGAN model: Integrated generator and discriminator training"""
    
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        
        # Metric tracking
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.psnr_metric = keras.metrics.Mean(name="PSNR")
        self.ssim_metric = keras.metrics.Mean(name="SSIM")
    
    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.psnr_metric, self.ssim_metric]
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, data):
        (x_water, x_dem), y_true = data
        
        # ===== 训练判别器 =====
        with tf.GradientTape() as tape:
            # 生成假图像
            y_fake = self.generator([x_water, x_dem], training=True)
            
            # 判别器预测
            real_output = self.discriminator([y_true, x_dem], training=True)
            fake_output = self.discriminator([y_fake, x_dem], training=True)
            
            # 判别器损失
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = real_loss + fake_loss
            
            # 混合精度缩放
            if isinstance(self.d_optimizer, mixed_precision.LossScaleOptimizer):
                scaled_d_loss = self.d_optimizer.get_scaled_loss(d_loss)
            else:
                scaled_d_loss = d_loss
        
        # 更新判别器
        if isinstance(self.d_optimizer, mixed_precision.LossScaleOptimizer):
            scaled_gradients = tape.gradient(scaled_d_loss, self.discriminator.trainable_variables)
            d_gradients = self.d_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # ===== 训练生成器 =====
        with tf.GradientTape() as tape:
            # 生成假图像
            y_fake = self.generator([x_water, x_dem], training=True)
            
            # 判别器预测
            fake_output = self.discriminator([y_fake, x_dem], training=True)
            
            # 生成器损失 = 内容损失 + 对抗损失
            content_loss = tf.reduce_mean(tf.abs(y_true - y_fake))  # MAE
            adversarial_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            
            g_loss = CONTENT_WEIGHT * content_loss + ADVERSARIAL_WEIGHT * adversarial_loss

            # 混合精度缩放
            if isinstance(self.g_optimizer, mixed_precision.LossScaleOptimizer):
                scaled_g_loss = self.g_optimizer.get_scaled_loss(g_loss)
            else:
                scaled_g_loss = g_loss
        
        # 更新生成器
        if isinstance(self.g_optimizer, mixed_precision.LossScaleOptimizer):
            scaled_gradients = tape.gradient(scaled_g_loss, self.generator.trainable_variables)
            g_gradients = self.g_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # 更新指标
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        self.psnr_metric.update_state(PSNR(y_true, y_fake))
        self.ssim_metric.update_state(SSIM(y_true, y_fake))
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (x_water, x_dem), y_true = data
        
        # 生成假图像
        y_fake = self.generator([x_water, x_dem], training=False)
        
        # 判别器预测
        real_output = self.discriminator([y_true, x_dem], training=False)
        fake_output = self.discriminator([y_fake, x_dem], training=False)
        
        # 计算损失
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        d_loss = real_loss + fake_loss
        
        content_loss = tf.reduce_mean(tf.abs(y_true - y_fake))
        adversarial_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        g_loss = CONTENT_WEIGHT * content_loss + ADVERSARIAL_WEIGHT * adversarial_loss
        
        # 更新指标
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        self.psnr_metric.update_state(PSNR(y_true, y_fake))
        self.ssim_metric.update_state(SSIM(y_true, y_fake))
        
        return {m.name: m.result() for m in self.metrics}


# 创建生成器和判别器
print("\n创建SRGAN模型...")
generator = make_generator(num_res_blocks=16, num_filters=64)
discriminator = make_discriminator(num_filters=64)

print("\n生成器架构：")
generator.summary()
print("\n判别器架构：")
discriminator.summary()

# 创建SRGAN模型
srgan = SRGAN(generator=generator, discriminator=discriminator)

#### 训练配置
g_optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LR_G, beta_1=0.9)
d_optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LR_D, beta_1=0.9)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=False) # 使用True可能更稳定，但这里为了匹配输出层用False

# 混合精度优化器包装
if policy.name == 'mixed_float16':
    g_optimizer = mixed_precision.LossScaleOptimizer(g_optimizer)
    d_optimizer = mixed_precision.LossScaleOptimizer(d_optimizer)
    print("  Optimizers wrapped with LossScaleOptimizer")

srgan.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer, loss_fn=loss_fn)

# 回调函数
# 自定义回调：仅保存生成器
class SaveGeneratorCallback(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_PSNR', mode='max', save_best_only=True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_metric = -float('inf') if mode == 'max' else float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == 'max':
            improved = current > self.best_metric
        else:
            improved = current < self.best_metric

        if improved or not self.save_best_only:
            self.best_metric = current
            # 访问SRGAN模型中的生成器并保存
            self.model.generator.save(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {current:.4f}, saving generator to {self.filepath}")

checkpoint_callback = SaveGeneratorCallback(
    filepath='out_save/srgan_best_model.h5',
    monitor='val_PSNR',
    mode='max',
    save_best_only=True
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_PSNR',
    mode='max',
    patience=ES_PATIENCE,
    min_delta=ES_MIN_DELTA,
    restore_best_weights=True,
    verbose=1
)

# 记录训练开始时间
start_time = time.time()

# 开始训练
print("\n开始训练SRGAN模型...")
history = srgan.fit(
    train_ds,
    epochs=TOTAL_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    callbacks=[checkpoint_callback, early_stopping]
)

# 保存训练历史
history_save_path = 'out_save/srgan_history.pkl'
os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
with open(history_save_path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# 计算训练总时间
end_time = time.time()
training_time = end_time - start_time

# 保存生成器（用于推理）
generator_save_path = 'out_save/srgan_generator.h5'
generator.save(generator_save_path)

print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

#### 可视化训练过程
fig = plt.figure(figsize=(16, 5))

# 生成器和判别器损失
plt.subplot(1, 4, 1)
plt.plot(history.history['g_loss'], label='Generator Loss', linewidth=2)
plt.plot(history.history['d_loss'], label='Discriminator Loss', linewidth=2)
plt.xlabel('Epochs', size=12)
plt.ylabel('Loss', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("GAN Losses")

# 验证损失
plt.subplot(1, 4, 2)
plt.plot(history.history['val_g_loss'], label='Val G Loss', linewidth=2)
plt.plot(history.history['val_d_loss'], label='Val D Loss', linewidth=2)
plt.xlabel('Epochs', size=12)
plt.ylabel('Loss', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Validation Losses")

# PSNR曲线
plt.subplot(1, 4, 3)
plt.plot(history.history['PSNR'], label='train PSNR', linewidth=2)
plt.plot(history.history['val_PSNR'], label='val PSNR', linewidth=2, linestyle='--')
plt.xlabel('Epochs', size=12)
plt.ylabel('PSNR (dB)', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("PSNR (SRGAN)")

# SSIM曲线
plt.subplot(1, 4, 4)
plt.plot(history.history['SSIM'], label='train SSIM', linewidth=2)
plt.plot(history.history['val_SSIM'], label='val SSIM', linewidth=2, linestyle='--')
plt.xlabel('Epochs', size=12)
plt.ylabel('SSIM', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("SSIM (SRGAN)")

plt.tight_layout()
plt.savefig('out_save/srgan_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n训练完成！模型和历史记录已保存。")
print("生成器模型：out_save/srgan_generator.h5")
print("完整模型：out_save/srgan_best_model.h5")
print("训练历史：out_save/srgan_history.pkl")


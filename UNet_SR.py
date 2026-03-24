# -*-coding:utf-8 -*-
"""
Author: UNet comparison model - DEM-guided water depth super-resolution
Date: December 6, 2025
Architecture: Classic UNet (Encoder-Decoder + Skip connections)
Input: Low-resolution water depth + High-resolution DEM
Output: High-resolution water depth 
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

# Set tf.data auto-tuning parameters
AUTOTUNE = tf.data.AUTOTUNE

# ============ Training configuration parameters ============
BATCH_SIZE = 4
TOTAL_EPOCHS = 200
STEPS_PER_EPOCH = 100
INITIAL_LR = 1e-4
ES_PATIENCE = 50
ES_MIN_DELTA = 0.1
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


#### UNet architecture components
def conv_block(x, filters, kernel_size=3, activation='relu', batch_norm=True):
    """Convolutional block: Conv -> BN -> Activation"""
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def encoder_block(x, filters):
    """Encoder block: 2 convolutions + MaxPooling"""
    skip = conv_block(x, filters)
    skip = conv_block(skip, filters)
    x = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return x, skip


def decoder_block(x, skip, filters):
    """Decoder block: UpSampling + Concat + 2 convolutions"""
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    return x


#### Custom UNet model class
class UNetModel(tf.keras.Model):
    """UNet model: Integrated PSNR/SSIM metric tracking"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="PSNR")
        self.ssim_metric = keras.metrics.Mean(name="SSIM")

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric, self.ssim_metric]

    def train_step(self, data):
        (x_water, x_dem), y = data

        with tf.GradientTape() as tape:
            y_pred = self([x_water, x_dem], training=True)
            loss = tf.reduce_mean(tf.abs(y - y_pred))  # MAE Loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(PSNR(y, y_pred))
        self.ssim_metric.update_state(SSIM(y, y_pred))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (x_water, x_dem), y = data
        y_pred = self([x_water, x_dem], training=False)
        loss = tf.reduce_mean(tf.abs(y - y_pred))

        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(PSNR(y, y_pred))
        self.ssim_metric.update_state(SSIM(y, y_pred))

        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        x_water, x_dem = data
        x_water = tf.cast(x_water, tf.float32)
        x_dem = tf.cast(x_dem, tf.float32)
        super_resolution_img = self([x_water, x_dem], training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        return super_resolution_img


#### Build UNet model
def make_unet_model(input_shape=(None, None, 1), base_filters=64):
    """
    UNet architecture: Encoder-Decoder + Skip connections
    Input: Low-resolution water depth (4m) + High-resolution DEM (1m)
    Output: High-resolution water depth (1m)
    """
    # ===== Input layer =====
    water_input = layers.Input(shape=input_shape, name='water_depth_4m')
    dem_input = layers.Input(shape=input_shape, name='dem_1m')
    
    # ===== Preprocessing =====
    water_x = layers.Rescaling(scale=1.0 / 255)(water_input)
    dem_x = layers.Rescaling(scale=1.0 / 255)(dem_input)
    
    # Upsample low-resolution water depth to 1m (4x)
    water_upsampled = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(water_x)
    
    # Concatenate water depth and DEM
    x = layers.Concatenate()([water_upsampled, dem_x])
    
    # ===== Encoder path =====
    # Level 1: 128x128 (assuming input is 128x128)
    x, skip1 = encoder_block(x, base_filters)
    
    # Level 2: 64x64
    x, skip2 = encoder_block(x, base_filters * 2)
    
    # Level 3: 32x32
    x, skip3 = encoder_block(x, base_filters * 4)
    
    # Level 4: 16x16
    x, skip4 = encoder_block(x, base_filters * 8)
    
    # ===== Bottleneck layer =====
    # Level 5: 8x8
    x = conv_block(x, base_filters * 16)
    x = conv_block(x, base_filters * 16)
    
    # ===== Decoder path =====
    # Level 4: 16x16
    x = decoder_block(x, skip4, base_filters * 8)
    
    # Level 3: 32x32
    x = decoder_block(x, skip3, base_filters * 4)
    
    # Level 2: 64x64
    x = decoder_block(x, skip2, base_filters * 2)
    
    # Level 1: 128x128
    x = decoder_block(x, skip1, base_filters)
    
    # ===== Output layer =====
    output = layers.Conv2D(1, 1, padding='same')(x)
    output = layers.Rescaling(scale=255)(output)
    
    return UNetModel([water_input, dem_input], output)


# Create model
print("\nCreating UNet model...")
model = make_unet_model(base_filters=64)
model.summary()

#### Training configuration
optim = keras.optimizers.Adam(learning_rate=INITIAL_LR)
model.compile(optimizer=optim, run_eagerly=False)

# Callback functions
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='out_save/unet_best_model.h5',
    monitor='val_PSNR',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_PSNR',
    mode='max',
    patience=ES_PATIENCE,
    min_delta=ES_MIN_DELTA,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_PSNR',
    mode='max',
    factor=0.5,
    patience=15,
    min_lr=1e-7,
    verbose=1
)

# Record training start time
start_time = time.time()

# Start training
print("\nStarting UNet model training...")
history = model.fit(
    train_ds,
    epochs=TOTAL_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr]
)

# Save training history
history_save_path = 'out_save/unet_history.pkl'
os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
with open(history_save_path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Calculate total training time
end_time = time.time()
training_time = end_time - start_time

# Save final model
model_save_path = 'out_save/unet_final_model.h5'
model.save(model_save_path)

print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

#### Visualize training process
fig = plt.figure(figsize=(15, 5))

# Loss curve
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='train loss', linewidth=2)
plt.plot(history.history['val_loss'], label='val loss', linewidth=2, linestyle='--')
plt.xlabel('Epochs', size=12)
plt.ylabel('Loss', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Loss (UNet)")

# PSNR curve
plt.subplot(1, 3, 2)
plt.plot(history.history['PSNR'], label='train PSNR', linewidth=2)
plt.plot(history.history['val_PSNR'], label='val PSNR', linewidth=2, linestyle='--')
plt.xlabel('Epochs', size=12)
plt.ylabel('PSNR (dB)', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("PSNR (UNet)")

# SSIM curve
plt.subplot(1, 3, 3)
plt.plot(history.history['SSIM'], label='train SSIM', linewidth=2)
plt.plot(history.history['val_SSIM'], label='val SSIM', linewidth=2, linestyle='--')
plt.xlabel('Epochs', size=12)
plt.ylabel('SSIM', size=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("SSIM (UNet)")

plt.tight_layout()
plt.savefig('out_save/unet_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining completed! Model and history records saved.")
print("Best model: out_save/unet_best_model_4m.h5")
print("Final model: out_save/unet_final_model_4m.h5")
print("Training history: out_save/unet_history_4m.pkl")


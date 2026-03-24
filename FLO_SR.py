# -*-coding:utf-8 -*-
"""
Author: XJQ
Date: August 04, 2025
"""
# Import necessary libraries
import numpy as np  # For numerical calculations and array operations
import matplotlib.pyplot as plt  # For data visualization
import glob  # For file path matching
import os  # For file and directory operations
import time  # For timing (training time statistics)
import tensorflow as tf  # For building and training deep learning models
import rasterio  # Added: For reading TIFF raster data
import pickle  # For saving training history
from tensorflow import keras  # Keras high-level API for fast model building
from tensorflow.keras import layers  # Keras layers for building network structure

# Set tf.data auto-tuning parameters for optimizing data loading performance
AUTOTUNE = tf.data.AUTOTUNE

# Check available GPU count to confirm GPU acceleration is used
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#### Dataset path settings
# High-resolution (HR) and low-resolution (LR) image folder paths - modified to TIFF file paths
train_hr_folder = "date/Label_1m_train/"  # Training set high-resolution image path
train_lr_folder = "date/Feature_4m_train/"  # Training set low-resolution image path
val_hr_folder = "date/Label_1m_val/"  # Validation set high-resolution image path
val_lr_folder = "date/Feature_4m_val/"  # Validation set low-resolution image path

# Create file path lists, limit sample count, filter for TIFF files
train_hr_files = sorted(
    [os.path.join(train_hr_folder, filename) for filename in os.listdir(train_hr_folder) if filename.endswith('.tif')])
train_lr_files = sorted(
    [os.path.join(train_lr_folder, filename) for filename in os.listdir(train_lr_folder) if filename.endswith('.tif')])
val_hr_files = sorted(
    [os.path.join(val_hr_folder, filename) for filename in os.listdir(val_hr_folder) if filename.endswith('.tif')])
val_lr_files = sorted(
    [os.path.join(val_lr_folder, filename) for filename in os.listdir(val_lr_folder) if filename.endswith('.tif')])


def load_image(lr_path, hr_path):
    # Key fix: Convert TensorFlow Tensor objects to Python strings
    lr_path_str = lr_path.numpy().decode('utf-8')
    hr_path_str = hr_path.numpy().decode('utf-8')

    # Use rasterio to read single-channel TIFF files
    with rasterio.open(lr_path_str) as src:
        lr_image = src.read(1)  # Read the first channel

    with rasterio.open(hr_path_str) as src:
        hr_image = src.read(1)  # Read the first channel

    # Expand dimensions, add channel dimension (from 2D to 3D: height, width, channel)
    lr_image = np.expand_dims(lr_image, axis=-1)
    hr_image = np.expand_dims(hr_image, axis=-1)

    # Convert to tensor
    lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
    hr_image = tf.convert_to_tensor(hr_image, dtype=tf.float32)

    return lr_image, hr_image  # Return (low-resolution image, high-resolution image) pair


# Create and preprocess training and validation sets
train_dataset = tf.data.Dataset.from_tensor_slices((train_lr_files, train_hr_files))
# Use tf.py_function to wrap custom loading function
train_dataset = train_dataset.map(lambda lr, hr: tf.py_function(
    load_image, [lr, hr], [tf.float32, tf.float32]),
                                  num_parallel_calls=AUTOTUNE)
train_cache = train_dataset.take(700).cache()  # Cache data to memory

# Validation set: same processing
valid_dataset = tf.data.Dataset.from_tensor_slices((val_lr_files, val_hr_files))
valid_dataset = valid_dataset.map(lambda lr, hr: tf.py_function(
    load_image, [lr, hr], [tf.float32, tf.float32]),
                                  num_parallel_calls=AUTOTUNE)
val_cache = valid_dataset.take(300).cache()

# Print dataset information to confirm sample count
print("Number of training samples:", len(train_hr_files))
print("Number of validation samples:", len(val_hr_files))

# Check image shapes in dataset (to confirm input/output dimensions match)
example_sample = next(iter(train_cache))  # Take one sample from training set
input_shape = example_sample[0].shape  # Low-resolution image shape
print("Input data shape:", input_shape)
label_shape = example_sample[1].shape  # High-resolution image shape
print("Label data shape:", label_shape)


#### Build tf.data.Dataset object (further optimize data loading process)
def dataset_object(dataset_cache, training=True):
    ds = dataset_cache  # Pass cached dataset

    # Set batch size to 16
    ds = ds.batch(16)

    if training:
        # Repeat dataset during training
        ds = ds.repeat()

    # Prefetch data
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# Create final dataset objects for training and validation sets
train_ds = dataset_object(train_cache, training=True)  # Training set (infinite repetition)
val_ds = dataset_object(val_cache, training=False)  # Validation set (no repetition)

#### Visualize input data pairs (check if low/high resolution images match)
lowres, highres = next(iter(train_ds))  # Take one batch of images from training set

# Display high-resolution images
plt.figure(figsize=(15, 15))
for i in range(min(7, len(highres))):
    ax = plt.subplot(1, 7, i + 1)
    # Remove channel dimension to correctly display single-channel images
    plt.imshow(tf.squeeze(highres[i]).numpy(), cmap='gray')
    plt.title(f"Highres\n{highres[i].shape}")
    plt.axis("off")
plt.show()

# Display corresponding low-resolution images
plt.figure(figsize=(15, 15))
for i in range(min(7, len(lowres))):
    ax = plt.subplot(1, 7, i + 1)
    plt.imshow(tf.squeeze(lowres[i]).numpy(), cmap='gray')
    plt.title(f"Lowres\n{lowres[i].shape}")
    plt.axis("off")
plt.show()


#### Define evaluation metrics (PSNR and SSIM for measuring super-resolution image quality)
def PSNR(high_resolution, super_resolution):
    # Calculate Peak Signal-to-Noise Ratio (PSNR)
    return tf.image.psnr(high_resolution, super_resolution, max_val=255)


def SSIM(high_resolution, super_resolution, max_val=255):
    # Calculate Structural Similarity Index (SSIM)
    high_resolution = tf.image.convert_image_dtype(high_resolution, tf.float32)
    super_resolution = tf.image.convert_image_dtype(super_resolution, tf.float32)

    # Calculate average SSIM value
    ssim_value = tf.image.ssim(high_resolution, super_resolution, max_val=max_val)
    return ssim_value


#### Define FLO-SR model (super-resolution model based on EDSR architecture)
class EDSRModel(tf.keras.Model):
    # Override training step (custom training logic)
    def train_step(self, data):
        x, y = data  # x: low-resolution input, y: high-resolution label

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward propagation
            # Calculate loss (MAE loss, configured in compile)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Calculate gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update evaluation metrics (PSNR, SSIM)
        self.compiled_metrics.update_state(y, y_pred)

        # Return metric results for current batch
        return {m.name: m.result() for m in self.metrics}

    # Override prediction step (custom prediction logic)
    def predict_step(self, x):
        # Expand dimension (add batch dimension) and convert to float32
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        # Model prediction (turn off training mode)
        super_resolution_img = self(x, training=False)
        # Clip pixel values to 0-255 range
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        # Round to integer pixel values
        # super_resolution_img = tf.round(super_resolution_img)
        # Remove batch dimension and convert to uint8 format
        super_resolution_img = tf.squeeze(
            tf.cast(super_resolution_img, tf.float32), axis=0
        )
        return super_resolution_img


# Define residual block (Residual Block)
def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)  # 64 3x3 convolution kernels, ReLU activation
    x = layers.Conv2D(64, 3, padding="same")(x)  # 64 3x3 convolution kernels, no activation
    x = layers.Add()([inputs, x])  # Skip connection: add input and output
    return x

# Define channel attention module (Channel Attention Module)
def ChannelAttention(inputs, reduction_ratio=16):
    # Global average pooling
    squeeze = layers.GlobalAveragePooling2D()(inputs)
    # Fully connected layer for dimensionality reduction
    squeeze = layers.Dense(inputs.shape[-1] // reduction_ratio, activation='relu')(squeeze)
    # Fully connected layer for dimensionality increase
    excitation = layers.Dense(inputs.shape[-1], activation='sigmoid')(squeeze)
    # Reshape to (batch_size, 1, 1, channels)
    excitation = layers.Reshape((1, 1, inputs.shape[-1]))(excitation)
    # Attention weighting
    scale = layers.Multiply()([inputs, excitation])
    return scale

# Build complete model - modified to 1-channel input/output
def make_model(num_filters, num_of_residual_blocks):
    # Input layer: accepts 1-channel images of any size
    input_layer = layers.Input(shape=(None, None, 1))
    # Normalize pixel values to [0,1] (divide by 255)
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)
    # Initial convolution: extract basic features
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)

    # Stack multiple residual blocks (16 here)
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)

    # Residual connection: add initial convolution output with stacked residual block results
    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])
    
    # Add global channel attention after residual block stacking
    x = ChannelAttention(x)

    # Upsampling: enlarge low-resolution feature maps to high-resolution (total 4x enlargement here)
    x = Upsampling(x)
    # Output layer: modified to 1 channel
    x = layers.Conv2D(1, 3, padding="same")(x)

    # Scale pixel values back to 0-255 range
    output_layer = layers.Rescaling(scale=255)(x)
    # Return custom EDSR model
    return EDSRModel(input_layer, output_layer)


# Define upsampling block (Upsampling Block)
def Upsampling(inputs, factor=2, **kwargs):
    # First pass through convolution to get feature maps with (factor^2) times channels
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    # Use depth_to_space for pixel rearrangement, enlarge by factor times
    x = tf.nn.depth_to_space(x, block_size=factor)
    # Another convolution and pixel rearrangement, total enlarge by factor^2 times
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    # # Another convolution and pixel rearrangement, total enlarge by factor^2 times
    # x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    # x = tf.nn.depth_to_space(x, block_size=factor)
    return x


# Build complete model - modified to 1-channel input/output
def make_model(num_filters, num_of_residual_blocks):
    # Input layer: accepts 1-channel images of any size
    input_layer = layers.Input(shape=(None, None, 1))
    # Normalize pixel values to [0,1] (divide by 255)
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)
    # Initial convolution: extract basic features
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)

    # Stack multiple residual blocks (16 here)
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)

    # Residual connection: add initial convolution output with stacked residual block results
    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])
    
    # Add global channel attention after residual block stacking
    x = ChannelAttention(x)

    # Upsampling: enlarge low-resolution feature maps to high-resolution (total 4x enlargement here)
    x = Upsampling(x)
    # Output layer: modified to 1 channel
    x = layers.Conv2D(1, 3, padding="same")(x)

    # Scale pixel values back to 0-255 range
    output_layer = layers.Rescaling(scale=255)(x)
    # Return custom EDSR model
    return EDSRModel(input_layer, output_layer)


# Create model instance: 64 filters, 16 residual blocks
model = make_model(num_filters=64, num_of_residual_blocks=16)

#### Model training configuration
# Define optimizer: Adam optimizer with dynamic learning rate adjustment
optim_edsr = keras.optimizers.Adam(
    learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]
    )
)

# Compile model: loss function is MAE (L1 loss), evaluation metrics are PSNR and SSIM
model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR, SSIM])

# Early stopping callback: prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=20,  # Tolerate 20 epochs without improvement
    restore_best_weights=True  # Restore best weights
)

# Record training start time
start_time = time.time()

# Start training
print("Starting model training...")
history = model.fit(train_ds, epochs=200, steps_per_epoch=100, validation_data=val_ds)

# Save training history
history_save_path = 'out_save/FLO_SR_4m.pkl'
os.makedirs(os.path.dirname(history_save_path), exist_ok=True)  # Create save directory
with open(history_save_path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Calculate total training time
end_time = time.time()
training_time = end_time - start_time

# Save model (H5 format)
model_save_path = 'out_save/FLO_SR_4m.h5'
model.save(model_save_path)

print("Training time: {:.2f} seconds".format(training_time))

#### Visualize training process (loss and PSNR curves)
# Plot training and validation loss curves
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], color='dodgerblue', label='train loss', linewidth=2.5, linestyle="--")
plt.plot(history.history['val_loss'], color='red', label='validation loss', linewidth=2.5)
plt.xlabel('Epochs', size=25)
plt.ylabel('Loss', size=25)
plt.legend(fontsize=20)
plt.grid(color='lightgrey', linestyle='-', linewidth=0.7)
plt.title("Train and Validation Loss", size=35)
plt.show()

# Plot training and validation PSNR curves
plt.figure(figsize=(10, 7))
plt.plot(history.history['PSNR'], color='green', label='train PSNR', linewidth=2.5, linestyle="--")
plt.plot(history.history['val_PSNR'], color='orange', label='validation PSNR', linewidth=2.5)
plt.xlabel('Epochs', size=25)
plt.ylabel('PSNR (dB)', size=25)
plt.legend(fontsize=20, loc='lower right')
plt.grid(color='lightgrey', linestyle='-', linewidth=0.7)
plt.title("Train and Validation PSNR", size=35)
plt.show()
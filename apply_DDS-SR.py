# -*-coding:utf-8 -*-
"""
Author:XJQ
Date: November 30, 2025
Function: Use trained dual-stream model for inference on test set
Updates:
- Fixed physics_informed_loss return value inconsistency
- Added individual loss trackers to match training script
- Added type conversion to ensure compatibility
- Support loading models trained with mixed precision
"""
import numpy as np
import tensorflow as tf
import rasterio
import os
import glob
import time
from tqdm import tqdm
from rasterio.transform import Affine
from tensorflow.keras import mixed_precision

# Set tf.data auto-tuning parameters
AUTOTUNE = tf.data.AUTOTUNE

# Custom learning rate scheduler (consistent with training script)
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler: Linear Warmup + Cosine annealing
    Note: Not needed during inference, but must be defined when loading model
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
    
    # Add operator overloading to support Keras internal operations
    def __mul__(self, other):
        """Support multiplication (needed for Keras history)"""
        return self
    
    def __rmul__(self, other):
        """Support reverse multiplication"""
        return self
    
    def __truediv__(self, other):
        """Support division"""
        return self
    
    def __add__(self, other):
        """Support addition"""
        return self


# Define evaluation metric functions
def PSNR(high_resolution, super_resolution):
    return tf.image.psnr(high_resolution, super_resolution, max_val=255)

def SSIM(high_resolution, super_resolution, max_val=255):
    high_resolution = tf.image.convert_image_dtype(high_resolution, tf.float32)
    super_resolution = tf.image.convert_image_dtype(super_resolution, tf.float32)
    ssim_value = tf.image.ssim(high_resolution, super_resolution, max_val=max_val)
    return ssim_value

# Physics-constrained loss function
def physics_informed_loss(y_true, y_pred, dem_hr, x_lr):
    """
    Physics-constrained loss = Reconstruction loss + Physics constraints + Volume consistency
    Returns: (Total loss, Individual loss dictionary) - Consistent with training script
    """
    # Ensure all tensor types are consistent
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dem_hr = tf.cast(dem_hr, tf.float32)
    x_lr = tf.cast(x_lr, tf.float32)
    
    # 1. Basic reconstruction loss (MAE)
    reconstruction_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # 2. Gradient consistency constraint
    dem_grad_x = dem_hr[:, :, 1:, :] - dem_hr[:, :, :-1, :]
    dem_grad_y = dem_hr[:, 1:, :, :] - dem_hr[:, :-1, :, :]
    
    water_grad_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    water_grad_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    
    gradient_consistency_x = tf.reduce_mean(tf.nn.relu(dem_grad_x * water_grad_x))
    gradient_consistency_y = tf.reduce_mean(tf.nn.relu(dem_grad_y * water_grad_y))
    gradient_consistency = gradient_consistency_x + gradient_consistency_y
    
    # 3. Local smoothness constraint
    smoothness_x = tf.reduce_mean(tf.square(water_grad_x))
    smoothness_y = tf.reduce_mean(tf.square(water_grad_y))
    smoothness_loss = smoothness_x + smoothness_y
    
    # 4. Water surface continuity constraint
    water_surface = dem_hr + y_pred
    surface_grad_x = water_surface[:, :, 1:, :] - water_surface[:, :, :-1, :]
    surface_grad_y = water_surface[:, 1:, :, :] - water_surface[:, :-1, :, :]
    surface_smoothness = tf.reduce_mean(tf.square(surface_grad_x)) + tf.reduce_mean(tf.square(surface_grad_y))
    
    # 5. Volume consistency constraint
    # 4x super-resolution, so ksize=4, strides=4
    y_pred_downsampled = tf.nn.avg_pool2d(y_pred, ksize=4, strides=4, padding='VALID')
    volume_consistency = tf.reduce_mean(tf.abs(y_pred_downsampled - x_lr))
    
    # Combined loss
    total_loss = reconstruction_loss + \
                 0.05 * gradient_consistency + \
                 0.001 * smoothness_loss + \
                 0.01 * surface_smoothness + \
                 0.1 * volume_consistency
    
    # Return total loss and individual loss dictionary (consistent with training script)
    loss_dict = {
        'recon': reconstruction_loss,
        'grad': gradient_consistency,
        'smooth': smoothness_loss,
        'surface': surface_smoothness,
        'volume': volume_consistency,
        'total': total_loss
    }
    
    return total_loss, loss_dict

# Define custom model class
class DualStreamEDSR(tf.keras.Model):
    """Dual-stream EDSR model: Integrated physics-constrained loss"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Basic metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.psnr_metric = tf.keras.metrics.Mean(name="PSNR")
        self.ssim_metric = tf.keras.metrics.Mean(name="SSIM")
        
        # Individual loss trackers (consistent with training script)
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.grad_loss_tracker = tf.keras.metrics.Mean(name="grad_loss")
        self.smooth_loss_tracker = tf.keras.metrics.Mean(name="smooth_loss")
        self.surface_loss_tracker = tf.keras.metrics.Mean(name="surface_loss")
        self.volume_loss_tracker = tf.keras.metrics.Mean(name="volume_loss")
    
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
            # Use physics-constrained loss, return total loss and individual losses
            loss, loss_dict = physics_informed_loss(y, y_pred, x_dem, x_water)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
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

# Define model loading function
def load_model_with_custom_objects(model_path):
    """
    Load custom model
    Note: Must provide custom objects consistent with training
    """
    custom_objects = {
        'DualStreamEDSR': DualStreamEDSR,
        'PSNR': PSNR,
        'SSIM': SSIM,
        'physics_informed_loss': physics_informed_loss,
        'WarmupCosineDecay': WarmupCosineDecay  # Add custom learning rate scheduler
    }
    
    print("  Custom objects registered:")
    for key in custom_objects.keys():
        print(f"    - {key}")
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# Helper function to get DEM path
def get_dem_path_from_water_path(water_path, dem_folder):
    """Get DEM path based on water depth file path"""
    filename = os.path.basename(water_path)
    dem_path = os.path.join(dem_folder, filename)
    
    if not os.path.exists(dem_path):
        base_name = filename.split('_')[0] if '_' in filename else filename.replace('.tif', '')
        dem_files = [f for f in os.listdir(dem_folder) if base_name in f and f.endswith('.tif')]
        if dem_files:
            dem_path = os.path.join(dem_folder, dem_files[0])
        else:
            print(f"Warning: No DEM found for {filename}, will use zeros")
            return None
    
    return dem_path

# Define single image prediction function
def predict_single_image_dual_stream(model, lr_water_path, hr_dem_path, output_path):
    """Predict single image using dual-stream model"""
    
    # Read low-resolution water depth
    with rasterio.open(lr_water_path) as src:
        lr_water = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
    
    # Read high-resolution DEM
    if hr_dem_path and os.path.exists(hr_dem_path):
        with rasterio.open(hr_dem_path) as src:
            hr_dem = src.read(1).astype(np.float32)
            # Normalize DEM
            hr_dem = (hr_dem - np.min(hr_dem)) / (np.max(hr_dem) - np.min(hr_dem) + 1e-8) * 255.0
    else:
        # If no DEM, use zero matrix (performance will degrade)
        hr_dem = np.zeros((height * 4, width * 4), dtype=np.float32)
        print(f"Warning: Using zero DEM for {os.path.basename(lr_water_path)}")
    
    # Expand dimensions
    lr_water = np.expand_dims(lr_water, axis=-1)
    lr_water = np.expand_dims(lr_water, axis=0)  # Batch dimension
    
    hr_dem = np.expand_dims(hr_dem, axis=-1)
    hr_dem = np.expand_dims(hr_dem, axis=0)  # Batch dimension
    
    # Convert to tensor
    lr_water_tensor = tf.convert_to_tensor(lr_water, dtype=tf.float32)
    hr_dem_tensor = tf.convert_to_tensor(hr_dem, dtype=tf.float32)
    
    # Predict (direct forward inference, avoid custom predict_step input unpacking issue)
    sr_tensor = model([lr_water_tensor, hr_dem_tensor], training=False)
    
    # Process prediction result
    sr_image = np.squeeze(sr_tensor)
    
    # Calculate upsampling factor
    upscale_factor = sr_image.shape[0] / height
    
    # Create new affine transformation parameters
    new_transform = Affine(
        transform.a / upscale_factor,
        transform.b, 
        transform.c, 
        transform.d, 
        transform.e / upscale_factor,
        transform.f 
    )
    
    # Update output image metadata
    profile.update(
        dtype=rasterio.float32,
        count=1,
        transform=new_transform,
        crs=crs,
        width=sr_image.shape[1],
        height=sr_image.shape[0]
    )
    
    # Save high-resolution image
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(sr_image.astype(rasterio.float32), 1)
    
    return sr_image

# Define batch processing function
def batch_process_dual_stream(model, lr_folder, dem_folder, output_folder):
    """Batch processing"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all tif files
    lr_files = sorted([f for f in glob.glob(os.path.join(lr_folder, '*.tif'))])
    
    start_time = time.time()
    
    # Batch processing
    for lr_file in tqdm(lr_files, desc="Processing with dual-stream model"):
        filename = os.path.basename(lr_file)
        output_path = os.path.join(output_folder, filename)
        
        # Get corresponding DEM file
        dem_path = get_dem_path_from_water_path(lr_file, dem_folder)
        
        try:
            predict_single_image_dual_stream(model, lr_file, dem_path, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\nBatch processing completed in {total_time:.2f} seconds")
    print(f"Total images processed: {len(lr_files)}")
    print(f"Average time per image: {total_time/len(lr_files):.2f} seconds")

# Main function
def main():
    """Main function"""
    # Configure paths
    model_path = 'out_save/DDS_SR_final_4x.h5'  # Use best model (consistent with training script)
    lr_folder = 'date/Feature_4m_test/'  # Test set low-resolution water depth
    dem_folder = 'date/DEM-01m/'  # High-resolution DEM
    output_folder = 'results/results_4m/DDS-SR-4m/'  # Output path
    
    print("=" * 70)
    print("Dual-stream model inference script - Updated to match training script")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    try:
        model = load_model_with_custom_objects(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("\nPossible reasons:")
        print("  1. Model file does not exist or path is incorrect")
        print("  2. Model was trained with mixed precision but current environment doesn't support it")
        print("  3. Custom object definitions don't match")
        import traceback
        traceback.print_exc()
        return
    
    # Display model structure
    print("\nModel architecture summary:")
    print("-" * 70)
    model.summary()
    print("-" * 70)
    
    # Execute batch processing
    print("\nStarting batch inference...")
    print(f"  Input path: {lr_folder}")
    print(f"  DEM path: {dem_folder}")
    print(f"  Output path: {output_folder}")
    print("-" * 70)
    
    batch_process_dual_stream(model, lr_folder, dem_folder, output_folder)
    
    print("\n" + "=" * 70)
    print("All inference tasks completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()


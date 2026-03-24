# -*-coding:utf-8 -*-
"""
UNet model inference script
Input: Low-resolution water depth + High-resolution DEM
Output: High-resolution water depth
"""
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
import tensorflow as tf
from tensorflow import keras

# Define UNetModel class (consistent with training script)
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

    # For inference, train_step and test_step are not required, but can keep empty shells or complete code for class compatibility
    # Here we only need it to be successfully loaded

# Load trained model
MODEL_PATH = 'out_save/unet_best_model.h5'
print(f"Loading model from {MODEL_PATH}...")
# Register custom objects
model = keras.models.load_model(MODEL_PATH, custom_objects={'UNetModel': UNetModel}, compile=False)
print("Model loaded successfully!")


def predict_single_image_unet(lr_water_path, hr_dem_path, output_path):
    """
    Perform super-resolution reconstruction of single image using UNet model
    
    Parameters:
        lr_water_path: Low-resolution water depth file path (4m)
        hr_dem_path: High-resolution DEM file path (1m)
        output_path: Output high-resolution water depth file path (1m)
    """
    # Read low-resolution water depth
    with rasterio.open(lr_water_path) as src:
        lr_water = src.read(1).astype(np.float32)
        lr_profile = src.profile
        lr_transform = src.transform
    
    # Read high-resolution DEM
    with rasterio.open(hr_dem_path) as src:
        hr_dem = src.read(1).astype(np.float32)
        hr_profile = src.profile
        hr_transform = src.transform
        hr_shape = (src.height, src.width)
    
    # DEM normalization
    hr_dem = (hr_dem - np.min(hr_dem)) / (np.max(hr_dem) - np.min(hr_dem) + 1e-8) * 255.0
    
    # Expand dimensions and convert to tensor
    lr_water_tensor = tf.expand_dims(tf.expand_dims(lr_water, axis=-1), axis=0)
    hr_dem_tensor = tf.expand_dims(tf.expand_dims(hr_dem, axis=-1), axis=0)
    
    # Model inference
    print(f"  Processing: {os.path.basename(lr_water_path)}")
    sr_tensor = model([lr_water_tensor, hr_dem_tensor], training=False)
    
    # Convert back to numpy and remove batch dimension
    sr_water = sr_tensor.numpy()[0, :, :, 0]
    sr_water = np.clip(sr_water, 0, 255)
    
    # Save result (using high-resolution geographic information)
    output_profile = hr_profile.copy()
    output_profile.update({
        'dtype': 'float32',
        'count': 1,
        'compress': 'lzw'
    })
    
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(sr_water.astype(np.float32), 1)
    
    print(f"  Saved to: {output_path}")
    
    # Calculate statistics
    print(f"  Output range: [{sr_water.min():.2f}, {sr_water.max():.2f}]")
    print(f"  Output mean: {sr_water.mean():.2f}")


def batch_process(lr_folder, dem_folder, output_folder):
    """
    Batch process all images in folder
    
    Parameters:
        lr_folder: Low-resolution water depth folder (4m)
        dem_folder: High-resolution DEM folder (1m)
        output_folder: Output folder
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all low-resolution water depth files
    lr_files = sorted([f for f in os.listdir(lr_folder) if f.endswith('.tif')])
    
    if len(lr_files) == 0:
        print(f"Error: No .tif files found in {lr_folder}")
        return
    
    print(f"Found {len(lr_files)} files to process.")
    
    for i, lr_filename in enumerate(lr_files, 1):
        print(f"\n[{i}/{len(lr_files)}]")
        
        lr_path = os.path.join(lr_folder, lr_filename)
        
        # Match corresponding DEM file
        dem_path = os.path.join(dem_folder, lr_filename)
        if not os.path.exists(dem_path):
            # Try other naming rules
            base_name = lr_filename.split('_')[0] if '_' in lr_filename else lr_filename.replace('.tif', '')
            dem_files = [f for f in os.listdir(dem_folder) if base_name in f and f.endswith('.tif')]
            if dem_files:
                dem_path = os.path.join(dem_folder, dem_files[0])
            else:
                print(f"  Warning: DEM file not found for {lr_filename}, skipping...")
                continue
        
        # Output path
        output_filename = lr_filename.replace('.tif', '.tif')
        output_path = os.path.join(output_folder, output_filename)
        
        # Execute inference
        try:
            predict_single_image_unet(lr_path, dem_path, output_path)
        except Exception as e:
            print(f"  Error processing {lr_filename}: {e}")
            continue
    
    print(f"\n✓ Batch processing completed! Results saved to: {output_folder}")


# ========== Main Program ==========
if __name__ == "__main__":
    # Method 1: Batch processing
    LR_FOLDER = "date/Feature_4m_test/"
    DEM_FOLDER = "date/DEM-01m/"
    OUTPUT_FOLDER = 'results/results_4m/Unet-SR-4m/'
    
    print("=" * 70)
    print("UNet Model Inference - Batch Processing")
    print("=" * 70)
    batch_process(LR_FOLDER, DEM_FOLDER, OUTPUT_FOLDER)
    
    # Method 2: Single file processing (example)
    # lr_path = "date/Feature_4m_test/sample.tif"
    # dem_path = "date/DEM-01m/sample.tif"
    # output_path = "out_save/sample_unet_sr.tif"
    # predict_single_image_unet(lr_path, dem_path, output_path)


# -*-coding:utf-8 -*-
"""
Author: XJQ
Date: August 04, 2025
Function: Use trained model to batch process low-resolution validation set 4m tif data, generate high-resolution 1m tif data
"""
# Import necessary libraries
import numpy as np
import tensorflow as tf
import rasterio
import os
import glob
import time
from tqdm import tqdm
from rasterio.transform import Affine

# Set tf.data auto-tuning parameters
AUTOTUNE = tf.data.AUTOTUNE

# Define evaluation metric functions (PSNR and SSIM) - Must be defined to match model structure
def PSNR(high_resolution, super_resolution):
    return tf.image.psnr(high_resolution, super_resolution, max_val=255)

def SSIM(high_resolution, super_resolution, max_val=255):
    high_resolution = tf.image.convert_image_dtype(high_resolution, tf.float32)
    super_resolution = tf.image.convert_image_dtype(super_resolution, tf.float32)
    ssim_value = tf.image.ssim(high_resolution, super_resolution, max_val=max_val)
    return ssim_value

# Define custom EDSRModel class - Must be defined to match model structure
class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    # Fix predict_step method, remove duplicate batch dimension expansion
    def predict_step(self, x):
        # Batch dimension already added externally, no need to add again here
        x = tf.cast(x, tf.float32)
        super_resolution_img = self(x, training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        # super_resolution_img = tf.round(super_resolution_img)
        super_resolution_img = tf.squeeze(tf.cast(super_resolution_img, tf.float32), axis=0)
        return super_resolution_img

# Define residual block and upsampling block - Must be defined to match model structure
def ResBlock(inputs):
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x

def Upsampling(inputs, factor=2, **kwargs):
    x = tf.keras.layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = tf.keras.layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    # x = tf.keras.layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    # x = tf.nn.depth_to_space(x, block_size=factor)
    return x

# Define model loading function
def load_model_with_custom_objects(model_path):
    # Create custom objects dictionary
    custom_objects = {
        'EDSRModel': EDSRModel,
        'PSNR': PSNR,
        'SSIM': SSIM,
        'ResBlock': ResBlock,
        'Upsampling': Upsampling
    }
    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# Define single image prediction function
def predict_single_image(model, lr_image_path, output_path):
    # Read low-resolution image and georeference information
    with rasterio.open(lr_image_path) as src:
        lr_image = src.read(1)  # Read first channel
        profile = src.profile.copy()  # Copy metadata information
        crs = src.crs  # Get coordinate reference system
        transform = src.transform  # Get affine transformation information
        width = src.width  # Low-resolution image width
        height = src.height  # Low-resolution image height
    
    # Expand dimensions, add channel dimension and batch dimension
    lr_image = np.expand_dims(lr_image, axis=-1)  # Add channel dimension
    lr_image = np.expand_dims(lr_image, axis=0)   # Add batch dimension
    
    # Convert to tensor and predict
    lr_tensor = tf.convert_to_tensor(lr_image, dtype=tf.float32)
    sr_tensor = model.predict(lr_tensor)
    
    # Process prediction result
    sr_image = np.squeeze(sr_tensor)  # Remove batch dimension and channel dimension
    
    # Calculate upsampling factor (high-resolution image size / low-resolution image size)
    upscale_factor = sr_image.shape[0] / height  # Should be 4x here
    
    # Create new affine transformation parameters, adjust pixel size from 4m to 1m
    # Keep top-left corner coordinates unchanged, but divide pixel width and height by scaling factor
    new_transform = Affine(
        transform.a / upscale_factor,  # Pixel width (becomes 1m)
        transform.b, 
        transform.c, 
        transform.d, 
        transform.e / upscale_factor,  # Pixel height (becomes 1m)
        transform.f 
    )
    
    # Update output image metadata
    profile.update(
        dtype=rasterio.float32,
        count=1,  # Output as single-channel image
        transform=new_transform,  # Use new affine transformation parameters
        crs=crs,
        width=sr_image.shape[1],  # Update width
        height=sr_image.shape[0]  # Update height
    )
    
    # Save high-resolution image
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(sr_image.astype(rasterio.float32), 1)
    
    return sr_image

# Define batch processing function
def batch_process(model, lr_folder, output_folder):
    # Create output folder (if not exists)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get paths of all tif files
    lr_files = sorted([f for f in glob.glob(os.path.join(lr_folder, '*.tif'))])
    if not lr_files:
        print(f"No tif files found in {lr_folder}. Please verify the LR folder path.")
        return
    
    # Record processing start time
    start_time = time.time()
    
    # Batch process each file
    for lr_file in tqdm(lr_files, desc="Processing images"):
        # Get filename
        filename = os.path.basename(lr_file)
        # Build output file path
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Predict and save high-resolution image
            predict_single_image(model, lr_file, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Calculate total processing time
    total_time = time.time() - start_time
    print(f"Batch processing completed in {total_time:.2f} seconds")
    print(f"Total images processed: {len(lr_files)}")
    print(f"Average time per image: {total_time/len(lr_files):.2f} seconds")

# Main function
def main():
    # Configure paths
    model_path = 'out_save/FLO_SR_4m.h5'  # Model save path
    lr_folder = 'date/Feature_4m_test/'  # Low-resolution validation set path
    output_folder = 'results/results_4m/FLO-SR-4m/'  # Output path

    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model_with_custom_objects(model_path)
    print("Model loaded successfully!")
    
    # Display model structure (optional)
    model.summary()
    
    # Execute batch processing
    print("Starting batch processing...")
    batch_process(model, lr_folder, output_folder)
    print("All processing tasks completed!")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Author: Assistant
Time: 2025/12/17
Function: Standardize raster files
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import re

# ================= Configuration Region =================
# Root directory
BASE_DIR = "esults/"

# Input folders
INPUT_FOLDERS = {
    "2m": "results_2m",
    "4m": "results_4m", 
    "8m": "results_8m"
}

# Output root directory
OUTPUT_ROOT = "results/normalized_results/"

# Normalization parameters
NORMALIZATION_PARAMS = {
    "_2": 2.2930,#Hydrodynamic model 2-year return period maximum depth
    "_50": 2.8730,#Hydrodynamic model 50-year return period maximum depth
}

# Threshold parameters
THRESHOLD = 0.05

# ================= Utility Functions =================

def normalize_raster_data(data, suffix):
    """
    Standardize raster data
    :param data: Raster data array
    :param suffix: Filename suffix ("_2" or "_50")
    :return: Standardized data
    """
    # Divide by 255
    normalized_data = data / 255.0
    
    # Multiply by corresponding coefficient based on suffix
    if suffix in NORMALIZATION_PARAMS:
        normalized_data = normalized_data * NORMALIZATION_PARAMS[suffix]
    
    # Set values below threshold to 0
    normalized_data[normalized_data < THRESHOLD] = 0
    
    return normalized_data

def process_tif_file(input_path, output_path):
    """
    Process single TIF file
    :param input_path: Input file path
    :param output_path: Output file path
    """
    try:
        # Read input file
        with rasterio.open(input_path) as src:
            # Read data
            data = src.read(1)  # Assume single band
            
            # Get filename and suffix
            filename = os.path.basename(input_path)
            # Use regex to extract suffix
            match = re.search(r'_([0-9]+)\.tif$', filename)
            if match:
                suffix = '_' + match.group(1)
            else:
                suffix = ''
            
            # Standardize data
            normalized_data = normalize_raster_data(data, suffix)
            
            # Create output directory (if not exists)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write output file
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=1,
                dtype=normalized_data.dtype,
                crs=src.crs,
                transform=src.transform,
                nodata=src.nodata
            ) as dst:
                dst.write(normalized_data, 1)
                
        print(f"Processed: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_all_folders():
    """
    Process all raster files in folders
    """
    print("Starting to process raster files...")
    
    total_processed = 0
    total_errors = 0
    
    # Iterate through each scale level folder
    for scale_key, input_folder in INPUT_FOLDERS.items():
        input_root = os.path.join(BASE_DIR, input_folder)
        output_root = os.path.join(OUTPUT_ROOT, input_folder)
        
        print(f"\nProcessing {scale_key} folder: {input_root}")
        
        # Iterate through all subfolders in input folder
        for root, dirs, files in os.walk(input_root):
            # Calculate relative path to maintain directory structure
            rel_path = os.path.relpath(root, input_root)
            
            # Iterate through all TIF files
            for file in files:
                if file.endswith('.tif'):
                    input_path = os.path.join(root, file)
                    # Build corresponding output path
                    output_path = os.path.join(output_root, rel_path, file)
                    
                    # Process file
                    if process_tif_file(input_path, output_path):
                        total_processed += 1
                    else:
                        total_errors += 1
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {total_processed} files")
    print(f"Error count: {total_errors} files")

# ================= Main Program =================
if __name__ == "__main__":
    process_all_folders()
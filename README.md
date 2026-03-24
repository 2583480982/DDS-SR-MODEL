# DDS-SR: Terrain-Guided Dual-Stream Physics-Constrained Water Depth Super-Resolution Model

## Project Introduction

This study proposes a terrain-guided dual-stream physics-constrained reconstruction framework for efficiently enhancing low-resolution water depth simulation results to high-resolution products. This framework employs parallel feature pathways to separately encode low-resolution water depth and high-resolution terrain information, and introduces cross-modal spatial attention to strengthen the modulation effect of terrain control regions on the reconstruction process; simultaneously, through physical soft constraints such as gradient consistency, water surface continuity, and local smoothness, it guides the output results to converge toward more reasonable hydrodynamic feasible domains.

## Key Features

- **Dual-Stream Architecture**: Parallel processing of low-resolution water depth and high-resolution DEM terrain information
- **Cross-Modal Spatial Attention**: Strengthening terrain's modulation effect on reconstruction process
- **Physics-Constrained Losses**:
  - Gradient consistency constraint
  - Water surface continuity constraint
  - Local smoothness constraint
- **Efficient Enhancement**: Enhancing 4m/8m resolution water depth data to 1m high-resolution

## Model Architecture

### Core Components

1. **Dual-Stream Encoder**
   - Water depth stream: Processing low-resolution water depth data
   - Terrain stream: Processing high-resolution DEM data

2. **Cross-Modal Spatial Attention Module**
   - Fusing water depth and terrain features
   - Strengthening the influence of terrain control regions

3. **Physics-Constrained Loss Functions**
   - Reconstruction loss (MAE)
   - Gradient consistency loss
   - Water surface continuity loss
   - Local smoothness loss

## File Structure

```
DDS_SR_MODEL/
├── DDS-SR.py                    # DDS-SR main model file
├── FLO_SR.py                    # FLO-SR comparison model
├── UNet_SR.py                   # UNet comparison model
├── SR_gan.py                    # SRGAN comparison model
├── apply_DDS-SR.py              # DDS-SR inference script
├── apply_FLO_SR.py              # FLO-SR inference script
├── apply_UNet_model.py          # UNet inference script
├── PLO_box.py                  # Model comparison visualization
├── plot_box2.py                # Ablation study visualization
├── raster_normalization.py        # Raster data standardization
└── DDS-SR-modularization/      # Modular code
    ├── config.py                # Configuration file
    ├── model.py                # Model definition
    ├── train.py                # Training script
    ├── losses.py               # Loss functions
    ├── data_loader.py          # Data loader
    ├── trainer.py             # Trainer
    ├── inference.py           # Inference script
    └── utils.py              # Utility functions
```

## Environment Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Rasterio
- scikit-image

See `environment.yml` for reference

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DDS-SR.git
cd DDS-SR

# Create and install environment
conda env create -f environment.yml
```

## Data Preparation

Training data requires the following folders:

```
date/
├── Label_1m_train/          # Training set high-resolution water depth (1m)
├── Feature_4m_train/        # Training set low-resolution water depth (4m)
├── DEM-01m/                # High-resolution DEM (1m)
├── Label_1m_val/            # Validation set high-resolution water depth
└── Feature_4m_val/          # Validation set low-resolution water depth
```

## Usage

### Training Model

```bash
# Train using modular code
cd DDS-SR-modularization
python train.py --config config.py
```

### Model Inference

```bash
# DDS-SR inference
python apply_DDS-SR.py --input_path input_folder --output_path output_folder

# FLO-SR inference
python apply_FLO_SR.py --input_path input_folder --output_path output_folder

# UNet inference
python apply_UNet_model.py --input_path input_folder --output_path output_folder
```

### Raster Data Standardization

```bash
# Standardize raster data, scaling 0-255 water depth values to real water depth range
python raster_normalization.py
```

### Result Visualization

```bash
# Model comparison box plots
python PLO_box.py

# Ablation study comparison
python plot_box2.py
```

## Training Configuration

Main hyperparameters are configured in `DDS-SR-modularization/config.py`:

```python
# Learning rate configuration
INITIAL_LR = 1e-4
MIN_LR = 1e-6
WARMUP_EPOCHS = 5

# Training parameters
TOTAL_EPOCHS = 50
BATCH_SIZE = 8
STEPS_PER_EPOCH = 100

# Physics constraint weights
GRAD_LOSS_WEIGHT = 0.1
SMOOTH_LOSS_WEIGHT = 0.01
SURFACE_LOSS_WEIGHT = 0.1
```

## Model Performance

Performance on test set:

| Model | PSNR (dB) | SSIM | RMSE (m) | MAE (m) |
|--------|-------------|--------|------------|-----------|
| LR (Baseline) | - | - | - | - |
| FLO-SR | - | - | - | - |
| UNet-SR | - | - | - | - |
| DDS-SR | - | - | - | - |

## Ablation Study

| Model Variant | PSNR (dB) | SSIM | Description |
|--------------|-------------|--------|-------------|
| DDS-No-Physics | - | - | Without physics constraints |
| DDS-SR | - | - | Full model |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Terrain-Guided Dual-Stream Physics-Constrained Water Depth Super-Resolution Reconstruction},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact via:

- Email: jq.xiao@hhu.edu.cn
- GitHub Issues: https://github.com/yourusername/DDS-SR/issues

## Acknowledgments

We thank all researchers and developers who have contributed to this project.

## Changelog

### v1.0.0 (2025-12-30)
- Initial release
- Implementation of DDS-SR core model
- Provided training and inference scripts
- Added comparison models (FLO-SR, UNet-SR, SRGAN)
- Implemented ablation study functionality

---

**Note**: This code is for academic research purposes only. For commercial use, please contact the author for authorization.

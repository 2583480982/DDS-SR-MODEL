"""
Loss function module
Define physics-informed loss and evaluation metrics
"""

import tensorflow as tf

from config import Config


def psnr_metric(y_true, y_pred, max_val=255.0):
    """
    Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        y_true: Ground truth
        y_pred: Prediction
        max_val: Maximum pixel value
        
    Returns:
        PSNR value
    """
    return tf.image.psnr(y_true, y_pred, max_val=max_val)


def ssim_metric(y_true, y_pred, max_val=255.0):
    """
    Structural Similarity Index (SSIM)
    
    Args:
        y_true: Ground truth
        y_pred: Prediction
        max_val: Maximum pixel value
        
    Returns:
        SSIM value
    """
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val=max_val)


def physics_informed_loss(y_true, y_pred, dem_hr, config=None):
    """
    Physics-informed loss function
    
    Physics constraints include:
    1. Reconstruction loss (MAE)
    2. Gradient consistency constraint (water depth should be shallow at high terrain)
    3. Local smoothness constraint
    4. Water surface continuity constraint (water surface elevation = DEM + water depth should be continuous)
    
    Args:
        y_true: True water depth
        y_pred: Predicted water depth
        dem_hr: High-resolution DEM
        config: Configuration object
        
    Returns:
        (Total loss, Individual loss dictionary)
    """
    if config is None:
        config = Config
    
    # Ensure all tensor types are consistent (mixed precision training)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dem_hr = tf.cast(dem_hr, tf.float32)
    
    # 1. Basic reconstruction loss (MAE)
    reconstruction_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Get ablation experiment mode
    ablation_mode = getattr(config, 'ABLATION_MODE', None)
    
    # Initialize physics loss terms
    gradient_consistency = tf.constant(0.0, dtype=tf.float32)
    smoothness_loss = tf.constant(0.0, dtype=tf.float32)
    surface_smoothness = tf.constant(0.0, dtype=tf.float32)
    
    # Only calculate physics loss in non-disabled modes
    if ablation_mode not in ['simple_concat', 'no_physics', 'baseline_single_stream']:
        # 2. Gradient consistency constraint
        # Calculate terrain gradient
        dem_grad_x = dem_hr[:, :, 1:, :] - dem_hr[:, :, :-1, :]
        dem_grad_y = dem_hr[:, 1:, :, :] - dem_hr[:, :-1, :, :]
        
        # Calculate water depth gradient
        water_grad_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        water_grad_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        
        # Gradient correlation: water depth should be shallow at high terrain (negative correlation)
        # Use ReLU to only penalize positive correlation cases
        gradient_consistency_x = tf.reduce_mean(tf.nn.relu(dem_grad_x * water_grad_x))
        gradient_consistency_y = tf.reduce_mean(tf.nn.relu(dem_grad_y * water_grad_y))
        gradient_consistency = gradient_consistency_x + gradient_consistency_y
        
        # 3. Local smoothness constraint
        smoothness_x = tf.reduce_mean(tf.square(water_grad_x))
        smoothness_y = tf.reduce_mean(tf.square(water_grad_y))
        smoothness_loss = smoothness_x + smoothness_y
        
        # 4. 水面连续性约束（水面高程 = DEM + 水深 应该连续）
        water_surface = dem_hr + y_pred
        surface_grad_x = water_surface[:, :, 1:, :] - water_surface[:, :, :-1, :]
        surface_grad_y = water_surface[:, 1:, :, :] - water_surface[:, :-1, :, :]
        surface_smoothness = tf.reduce_mean(tf.square(surface_grad_x)) + tf.reduce_mean(tf.square(surface_grad_y))
    
    # 综合损失（使用配置中的权重）
    w_recon = config.LOSS_WEIGHT_RECONSTRUCTION
    w_grad = config.LOSS_WEIGHT_GRADIENT
    w_smooth = config.LOSS_WEIGHT_SMOOTHNESS
    w_surf = config.LOSS_WEIGHT_SURFACE
    
    if ablation_mode in ['simple_concat', 'no_physics', 'baseline_single_stream']:
        w_grad = 0.0
        w_smooth = 0.0
        w_surf = 0.0
    
    total_loss = (
        w_recon * reconstruction_loss +
        w_grad * gradient_consistency +
        w_smooth * smoothness_loss +
        w_surf * surface_smoothness
    )
    
    # 返回总损失和各项损失字典
    loss_dict = {
        'recon': reconstruction_loss,
        'grad': gradient_consistency,
        'smooth': smoothness_loss,
        'surface': surface_smoothness,
        'total': total_loss
    }
    
    return total_loss, loss_dict


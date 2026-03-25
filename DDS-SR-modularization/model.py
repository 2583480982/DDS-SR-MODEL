"""
Model definition module
Contains dual-stream EDSR model and all network components
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

from config import Config
from losses import physics_informed_loss, psnr_metric, ssim_metric


def ResBlock(inputs):
    """
    Residual block
    
    Args:
        inputs: Input tensor
        
    Returns:
        Residual block output
    """
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


def ChannelAttention(inputs, reduction_ratio=16):
    """
    Channel attention module
    
    Args:
        inputs: Input tensor
        reduction_ratio: Dimension reduction ratio
        
    Returns:
        Attention-weighted features
    """
    channel = inputs.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(inputs)
    squeeze = layers.Dense(channel // reduction_ratio, activation='relu')(squeeze)
    excitation = layers.Dense(channel, activation='sigmoid')(squeeze)
    excitation = layers.Reshape((1, 1, channel))(excitation)
    scale = layers.Multiply()([inputs, excitation])
    return scale


def CrossModalAttention(water_features, dem_features):
    """
    跨模态注意力机制：使用DEM特征引导水深特征
    
    Args:
        water_features: 水深特征
        dem_features: DEM特征
        
    Returns:
        融合后的特征
    """
    # DEM生成空间注意力图
    spatial_attention = layers.Conv2D(64, 3, padding="same", activation='relu')(dem_features)
    spatial_attention = layers.Conv2D(1, 1, activation='sigmoid')(spatial_attention)
    
    # 应用注意力到水深特征
    water_attended = layers.Multiply()([water_features, spatial_attention])
    
    # 特征融合
    concat = layers.Concatenate()([water_attended, dem_features])
    fused = layers.Conv2D(64, 3, padding="same", activation='relu')(concat)
    fused = layers.Conv2D(64, 3, padding="same")(fused)
    
    return fused


def Upsampling(inputs, factor=2, **kwargs):
    """
    上采样块（使用Sub-pixel Convolution）
    
    Args:
        inputs: 输入tensor
        factor: 上采样倍数
        **kwargs: 其他Conv2D参数
        
    Returns:
        上采样后的tensor
    """
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


class DualStreamEDSR(keras.Model):
    """
    双流EDSR模型
    集成物理约束损失的自定义训练步骤
    """
    
    def __init__(self, data_max_value=255.0, *args, **kwargs):
        """初始化模型和指标"""
        super().__init__(*args, **kwargs)
        
        self.data_max_value = data_max_value
        
        # 基础指标
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="PSNR")
        self.ssim_metric = keras.metrics.Mean(name="SSIM")
        
        # 各项损失跟踪器
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")
        self.smooth_loss_tracker = keras.metrics.Mean(name="smooth_loss")
        self.surface_loss_tracker = keras.metrics.Mean(name="surface_loss")
    
    @property
    def metrics(self):
        """返回所有指标"""
        return [
            self.loss_tracker,
            self.psnr_metric,
            self.ssim_metric,
            self.recon_loss_tracker,
            self.grad_loss_tracker,
            self.smooth_loss_tracker,
            self.surface_loss_tracker
        ]
    
    def train_step(self, data):
        """
        自定义训练步骤
        
        Args:
            data: ((x_water, x_dem), y)
            
        Returns:
            指标字典
        """
        (x_water, x_dem), y = data
        
        with tf.GradientTape() as tape:
            # 前向传播
            y_pred = self([x_water, x_dem], training=True)
            
            # 计算物理约束损失
            loss, loss_dict = physics_informed_loss(y, y_pred, x_dem)
            
            # 混合精度训练：缩放损失
            if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss
        
        # 计算梯度
        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        
        # 混合精度训练：反缩放梯度
        if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients
        
        # 应用梯度
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 更新指标
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr_metric(y, y_pred, max_val=self.data_max_value))
        self.ssim_metric.update_state(ssim_metric(y, y_pred, max_val=self.data_max_value))
        self.recon_loss_tracker.update_state(loss_dict['recon'])
        self.grad_loss_tracker.update_state(loss_dict['grad'])
        self.smooth_loss_tracker.update_state(loss_dict['smooth'])
        self.surface_loss_tracker.update_state(loss_dict['surface'])
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        自定义验证步骤
        
        Args:
            data: ((x_water, x_dem), y)
            
        Returns:
            指标字典
        """
        (x_water, x_dem), y = data
        
        # 前向传播
        y_pred = self([x_water, x_dem], training=False)
        
        # 计算损失
        loss, loss_dict = physics_informed_loss(y, y_pred, x_dem)
        
        # 更新指标
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr_metric(y, y_pred, max_val=self.data_max_value))
        self.ssim_metric.update_state(ssim_metric(y, y_pred, max_val=self.data_max_value))
        self.recon_loss_tracker.update_state(loss_dict['recon'])
        self.grad_loss_tracker.update_state(loss_dict['grad'])
        self.smooth_loss_tracker.update_state(loss_dict['smooth'])
        self.surface_loss_tracker.update_state(loss_dict['surface'])
        
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        """
        自定义预测步骤
        
        Args:
            data: 输入数据
            
        Returns:
            预测的高分辨率水深
        """
        if isinstance(data, (list, tuple)):
            if len(data) == 2:
                x_water, x_dem = data[0], data[1]
            elif len(data) == 1 and isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
                x_water, x_dem = data[0]
            else:
                raise ValueError(f"无法解析输入数据格式，期望2个输入，得到: {len(data)}")
        elif isinstance(data, dict):
            x_water = data.get('water_depth_4m', None)
            x_dem = data.get('dem_1m', None)
            if x_water is None or x_dem is None:
                raise ValueError("字典格式缺少必要的键: 'water_depth_4m' 或 'dem_1m'")
        else:
            try:
                if hasattr(data, '__len__') and len(data) == 2:
                    x_water, x_dem = data[0], data[1]
                else:
                    raise ValueError(f"无法解析输入数据格式: {type(data)}")
            except (ValueError, TypeError, AttributeError) as e:
                raise ValueError(f"无法解析输入数据格式: {type(data)}, 错误: {e}")
        
        # 确保是tensor格式
        x_water = tf.cast(x_water, tf.float32)
        x_dem = tf.cast(x_dem, tf.float32)
        
        # 调用模型
        super_resolution_img = self([x_water, x_dem], training=False)
        # 裁剪到合法范围
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, self.data_max_value)
        return super_resolution_img


def build_dual_stream_model(num_filters=64, num_residual_blocks=16, use_mixed_precision=True, data_max_value=255.0, upscale_factor=4, ablation_mode=None):
    """
    构建双流EDSR模型
    
    Args:
        num_filters: 卷积核数量
        num_residual_blocks: 残差块总数量
        use_mixed_precision: 是否使用混合精度
        data_max_value: 数据最大值（用于归一化）
        upscale_factor: 上采样倍数
        ablation_mode: 消融实验模式
        
    Returns:
        DualStreamEDSR模型实例
    """
    import math
    
    # ==================== 输入层 ====================
    water_input = layers.Input(shape=(None, None, 1), name='water_depth_4m')
    dem_input = layers.Input(shape=(None, None, 1), name='dem_1m')
    
    # ==================== 水深流（低分辨率） ====================
    # 根据data_max_value进行归一化
    water_x = layers.Rescaling(scale=1.0 / data_max_value)(water_input)
    water_x = layers.Conv2D(num_filters, 3, padding="same")(water_x)
    water_features = water_x
    
    # 水深残差块
    for _ in range(num_residual_blocks // 2):
        water_features = ResBlock(water_features)
    
    water_features = layers.Conv2D(num_filters, 3, padding="same")(water_features)
    water_features = layers.Add()([water_x, water_features])
    
    # Baseline模式不使用通道注意力
    if ablation_mode != 'baseline_single_stream':
        water_features = ChannelAttention(water_features)
    
    # ==================== DEM流（高分辨率） ====================
    if ablation_mode != 'baseline_single_stream':
        # 双流模式：构建DEM流
        # 根据data_max_value进行归一化
        dem_x = layers.Rescaling(scale=1.0 / data_max_value)(dem_input)
        dem_x = layers.Conv2D(num_filters, 3, padding="same")(dem_x)
        dem_features = dem_x
        
        # DEM残差块
        for _ in range(num_residual_blocks // 2):
            dem_features = ResBlock(dem_features)
        
        dem_features = layers.Conv2D(num_filters, 3, padding="same")(dem_features)
        dem_features = layers.Add()([dem_x, dem_features])
        dem_features = ChannelAttention(dem_features)
    
    # ==================== 上采样水深特征到1m分辨率 ====================
    # 根据 upscale_factor 动态确定上采样策略
    # 如果是2的幂次（2, 4, 8...），使用多个2x上采样块堆叠（EDSR标准做法）
    # 否则直接使用单个上采样块
    if upscale_factor & (upscale_factor - 1) == 0:
        num_upsample_blocks = int(math.log2(upscale_factor))
        x = water_features
        for _ in range(num_upsample_blocks):
            x = Upsampling(x, factor=2)
        water_upsampled = x
    else:
        water_upsampled = Upsampling(water_features, factor=upscale_factor)
    
    # ==================== 跨模态注意力融合 ====================
    # 根据消融实验模式选择融合方式
    if ablation_mode == 'baseline_single_stream':
        # 单流模式：直接使用上采样后的水深特征，跳过融合
        fused_features = water_upsampled
    elif ablation_mode in ['simple_concat', 'no_attention']:
        # 简单拼接：不使用跨模态注意力，直接拼接后融合
        # 降级：仅使用1x1卷积进行简单融合，去除空间上下文感知能力
        concat = layers.Concatenate()([water_upsampled, dem_features])
        fused_features = layers.Conv2D(num_filters, 1, padding="same")(concat)
    else:
        # 默认：使用跨模态注意力融合
        fused_features = CrossModalAttention(water_upsampled, dem_features)
    
    # ==================== 残差细化 ====================
    for _ in range(num_residual_blocks // 2):
        fused_features = ResBlock(fused_features)
    
    # ==================== 输出层 ====================
    output = layers.Conv2D(1, 3, padding="same")(fused_features)
    # 反归一化到原始范围
    output = layers.Rescaling(scale=data_max_value)(output)
    
    # 混合精度训练：强制输出为float32
    if use_mixed_precision:
        output = layers.Activation('linear', dtype='float32')(output)
    
    # 创建模型，传入data_max_value
    model = DualStreamEDSR(data_max_value=data_max_value, inputs=[water_input, dem_input], outputs=output)
    
    return model

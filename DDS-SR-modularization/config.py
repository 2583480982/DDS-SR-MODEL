"""
Configuration file
Define all hyperparameters for training and inference
"""

class Config:
    """Global configuration for training and inference"""
    
    # ==================== Data Path Configuration ====================
    # Training set paths
    TRAIN_HR_WATER_FOLDER = "date/Label_1m_train/"     # High-resolution water depth
    TRAIN_LR_WATER_FOLDER = "date/Feature_4m_train/"   # Low-resolution water depth
    TRAIN_HR_DEM_FOLDER = "date/DEM-01m/"              # High-resolution DEM
    
    # Validation set paths
    VAL_HR_WATER_FOLDER = "date/Label_1m_val/"
    VAL_LR_WATER_FOLDER = "date/Feature_4m_val/"
    VAL_HR_DEM_FOLDER = "date/DEM-01m/"

    # Test set paths
    TEST_LR_WATER_FOLDER = "date/Feature_4m_test/"     # Test set low-resolution water depth
    TEST_HR_DEM_FOLDER = "date/DEM-01m/"               # Test set DEM
    
    # Output paths
    OUTPUT_DIR = "out_save/"
    MODEL_SAVE_PATH = "out_save/DDS_SR_best_4x_F.h5"
    FINAL_MODEL_PATH = "out_save/DDS_SR_final_4x_F.h5"
    HISTORY_SAVE_PATH = "out_save/training_history_DDS_SR_4x_F.pkl"
    
    # ==================== Training Hyperparameters ====================
    # Learning rate configuration
    USE_PIECEWISE_LR = True       # Keep consistent with original script, use WarmupCosineDecay
    INITIAL_LR = 1e-4              # Initial learning rate
    MIN_LR = 1e-6                  # Minimum learning rate (for cosine annealing)
    WARMUP_EPOCHS = 5              # Warmup epochs (for cosine annealing)
    LR_BOUNDARIES = [5000]         # Learning rate change boundaries (for PiecewiseConstant)
    LR_VALUES = [1e-4, 5e-5]       # Learning rate value list (for PiecewiseConstant)
    
    TOTAL_EPOCHS = 50             # Total training epochs
    STEPS_PER_EPOCH = 100          # Steps per epoch
    
    # Batch size
    BATCH_SIZE = 8                # Adjust based on GPU memory [4/8/16/32]
    
    # Random seed (critical: ensure reproducibility)
    RANDOM_SEED = 42               # Random seed
    
    # Early Stopping configuration
    ES_PATIENCE = 50               # Patience value
    ES_MIN_DELTA = 0.1             # PSNR minimum improvement
    ES_MONITOR = 'val_PSNR'        # 监控指标
    ES_MODE = 'max'                # 指标模式（max/min）
    
    # ==================== 模型参数 ====================
    NUM_FILTERS = 64               # 卷积核数量
    NUM_RESIDUAL_BLOCKS = 16       # 残差块数量
    UPSCALE_FACTOR = 4             # 上采样倍数 (4m -> 1m)
    
    # 关键修改：数据最大值（用于归一化和反归一化）
    # 如果使用原始水深数据，请设置为数据的最大可能值（例如实际最大水深）
    # 原始代码默认为 255.0
    DATA_RANGE_MAX = 255.0         
    
    # ==================== 损失函数权重 ====================
    # 修正：恢复原始脚本的权重参数
    LOSS_WEIGHT_RECONSTRUCTION = 1.0     # 重建损失权重
    LOSS_WEIGHT_GRADIENT = 0.08          # 梯度一致性权重
    LOSS_WEIGHT_SMOOTHNESS = 0.003       # 平滑性权重
    LOSS_WEIGHT_SURFACE = 0.015           # 水面连续性权重
    
    # ==================== 混合精度训练 ====================
    # 修正：恢复原始脚本的混合精度设置
    USE_MIXED_PRECISION = True      # 是否启用混合精度（需要支持的GPU）
    
    # ==================== 推理配置 ====================
    INFERENCE_BATCH_SIZE = 1       # 推理时的批次大小
    OUTPUT_PRED_FOLDER = "DDS_SR_test_best_4x_F/"  # 预测结果保存路径
    SAVE_AS_GEOTIFF = True         # 是否保存为GeoTIFF格式
    
    # ==================== 数据处理 ====================
    CACHE_TRAIN_SIZE = 2250         # 训练集缓存大小
    CACHE_VAL_SIZE = 750          # 验证集缓存大小
    
    # ==================== 其他配置 ====================
    VERBOSE = 1                    # 日志详细程度
    
    # ==================== 对比实验模式 ====================
    # 当与FLO进行对比时，建议启用此模式
    COMPARISON_MODE = False        # 对比实验模式（使用纯MAE损失，关闭物理约束）
    
    # ==================== 消融实验模式 ====================
    # 可选值: 
    # None: 完整模型（默认）
    # 'simple_concat': 1. DEM+简单拼接（没有跨模态注意力和物理损失）
    # 'no_physics': 2. 其余都有没有物理约束
    # 'no_attention': 3. 其余都有没有跨模态注意力
    # 'baseline_single_stream': 4. Baseline（单流EDSR）（没有DEM流、跨模态注意力和物理损失）
    ABLATION_MODE = None

    @staticmethod
    def print_config():
        """打印配置信息"""
        print("="*60)
        print("配置信息")
        print("="*60)
        print(f"消融实验模式: {Config.ABLATION_MODE}")
        print(f"总训练轮数: {Config.TOTAL_EPOCHS}")
        print(f"批次大小: {Config.BATCH_SIZE}")
        print(f"数据范围最大值: {Config.DATA_RANGE_MAX}")
        print(f"混合精度训练: {'启用' if Config.USE_MIXED_PRECISION else '禁用'}")
        print(f"损失权重: Grad={Config.LOSS_WEIGHT_GRADIENT}, Smooth={Config.LOSS_WEIGHT_SMOOTHNESS}, Surf={Config.LOSS_WEIGHT_SURFACE}")
        print("="*60)

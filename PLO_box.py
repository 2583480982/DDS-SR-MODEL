# -*- coding: utf-8 -*-
"""
Author :XJQ
Time :2025/12/17 19:44
Function:
"""
import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ================= 配置区域 =================
# 根目录路径 (请修改为你实际的路径)
BASE_DIR = "results//normalized_results//"

# 阈值设置：水深大于此值才被视为“有水”，用于计算命中率和误报率
WATER_THRESHOLD = 0.05

# 模型列表与文件夹名称映射
# 格式: {"图表中显示的名字": "文件夹名前缀"}
# 注意：代码会自动拼接后缀，如 'DDS-SR' -> 'DDS-SR-2m'
MODEL_MAPPING = {
    "LR": "LR",
    "FLO-SR": "FLO-SR",
    # "SRGAN": "Srgan",  # 根据你的目录可能需要调整大小写
    "UNet-SR": "Unet-SR",  # 注意检查目录是 Unet-SR 还是 Unet_SR
    "DDS-SR": "DDS-SR"
}

# 颜色板：为论文优化的一组高区分度配色（手动指定，避免默认样式）
# 顺序将与 hue_order 中的模型顺序一致
# PALETTE = [
#     "#7C9895",  # LR       - 蓝色
#     "#C9DCC4",  # FLO-SR   - 橙色
#     "#DAA87C",  # UNet-SR  - 绿色
#     "#F4EEAC",  # DDS-SR   - 红色
# ]
PALETTE = [
    "#ECA8A9",  # LR       - 蓝色
    "#74AED4",  # FLO-SR   - 橙色
    "#D3E2B7",  # UNet-SR  - 绿色
    "#CFAFD4",  # DDS-SR   - 红色
]

# ================= 核心函数 =================

def read_geotiff(path):
    """Read GeoTIFF and handle invalid values"""
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            data = np.nan_to_num(data, nan=0.0)
            data[data < 0] = 0
            return data
    except Exception as e:
        return None


def calculate_metrics(pred, gt):
    """Calculate 6 metrics for single image"""
    # 1. Basic error metrics
    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))

    # 2. Image quality metrics
    # data_range set to max value of gt, or fixed physical upper limit (e.g., 10m)
    # Here dynamically obtained, if all 0 then set to 1 to prevent error
    d_range = max(np.max(gt), 0.001)
    try:
        val_psnr = psnr(gt, pred, data_range=d_range)
    except:
        val_psnr = 0

    try:
        val_ssim = ssim(gt, pred, data_range=d_range)
    except:
        val_ssim = 0

    # 3. Binary classification metrics (flooded/not flooded)
    pred_mask = pred > WATER_THRESHOLD
    gt_mask = gt > WATER_THRESHOLD

    # Confusion matrix elements
    TP = np.sum(pred_mask & gt_mask)
    FP = np.sum(pred_mask & ~gt_mask)
    FN = np.sum(~pred_mask & gt_mask)

    # Hit Rate (POD) = TP / (TP + FN)
    hit_rate = TP / (TP + FN) if (TP + FN) > 0 else 1.0  # If GT has no water, default to hit

    # False Alarm Ratio = FP / (TP + FP)
    # Note: Academically FAR sometimes refers to FP/(FP+TN), but in flood field often refers to False Discovery Rate (FP / Predicted Positive)
    # Here adopt FP / (TP + FP) to reflect how many predicted waters are false
    far = FP / (TP + FP) if (TP + FP) > 0 else 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "PSNR": val_psnr,
        "SSIM": val_ssim,
        "Hit Rate": hit_rate,
        "FAR": far
    }


def collect_data():
    """Traverse folders to collect all data"""
    records = []
    scales = ["2m", "4m", "8m"]

    print("Starting data processing, may take a few minutes, please wait...")

    for scale in scales:
        scale_dir = os.path.join(BASE_DIR, f"results_{scale}")

        # Determine GT directory (based on your description, 8m contains HR-1m, others may be HR-val-1m, or unified)
        # Here try to automatically find
        gt_candidates = glob.glob(os.path.join(scale_dir, "HR*1m"))
        if not gt_candidates:
            print(f"Warning: No GT folder found in {scale_dir}")
            continue
        gt_dir = gt_candidates[0]

        # Get all GT file names
        gt_files = [os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.tif"))]

        for display_name, folder_prefix in MODEL_MAPPING.items():
            # Construct model folder name, e.g., FLO-SR-2m
            model_folder_name = f"{folder_prefix}-{scale}"
            model_dir = os.path.join(scale_dir, model_folder_name)

            # Handle possible Unet_SR vs Unet-SR naming inconsistency
            if not os.path.exists(model_dir) and "Unet" in folder_prefix:
                model_dir = os.path.join(scale_dir, f"{folder_prefix.replace('-', '_')}_{scale}")

            if not os.path.exists(model_dir):
                print(f"  Skipping {display_name} in {scale}: Folder not found")
                continue

            print(f"  Processing {scale} - {display_name}...")

            for fname in gt_files:
                gt_path = os.path.join(gt_dir, fname)
                pred_path = os.path.join(model_dir, fname)

                if os.path.exists(pred_path):
                    gt_data = read_geotiff(gt_path)
                    pred_data = read_geotiff(pred_path)

                    if gt_data is not None and pred_data is not None:
                        metrics = calculate_metrics(pred_data, gt_data)
                        metrics["Scale"] = scale
                        metrics["Model"] = display_name
                        metrics["Sample"] = fname  # 记录样本文件名
                        records.append(metrics)

    return pd.DataFrame(records)


def plot_boxplots(df):
    """Draw 2x3 box plots"""
    metrics_list = ["PSNR(db)", "SSIM", "RMSE(m)", "MAE(m)", "Hit Rate", "FAR"]
    metrics_list_dict ={"PSNR(db)":"PSNR","SSIM":"SSIM","RMSE(m)":"RMSE","MAE(m)":"MAE","Hit Rate":"Hit Rate","FAR":"FAR"}
    # Set global plotting style
    sns.set_theme(style="ticks", font="Times New Roman", font_scale=1.1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    abc = ["(a)","(b)","(c)","(d)","(e)","(f)"]

    for i, metric in enumerate(metrics_list):
        ax = axes[i]

        # Draw box plot
        sns.boxplot(
            data=df,
            x="Scale",
            y=metrics_list_dict[metric],
            hue="Model",
            hue_order=list(MODEL_MAPPING.keys()),
            ax=ax,
            palette=PALETTE[:len(MODEL_MAPPING)],
            width=0.7,
            linewidth=1.2,
            fliersize=2,  # Outlier size, set to 0 to hide
            showfliers=False  # Paper charts usually hide too many outliers to keep clean, or set to True
        )

        # Beautify
        ax.set_title(f"{abc[i]} {metric} for Each Model", fontweight='bold', fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(metric,fontweight='bold', fontsize=16)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        # Set x/y axis tick font size (core supplement)
        ax.tick_params(axis='x', labelsize=16)  # Set x-axis ticks separately
        ax.tick_params(axis='y', labelsize=16)  # Set y-axis ticks separately

        # Show legend in each subplot, automatically find best position
        if ax.get_legend():
            ax.legend(loc='best', fontsize=14, frameon=True, fancybox=True, shadow=False)

    plt.tight_layout()

    save_path = "figs/model_comparison_boxplot3.png"
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")


# ================= Main Program =================

if __name__ == "__main__":
    # 1. Collect data
    df = collect_data()

    if not df.empty:
        # 2. Export detailed metrics table for each sample
        detail_csv_path = "figs/sample_detail_results.csv"
        # Reorder columns to make table more readable
        column_order = ['Scale', 'Model', 'Sample', 'PSNR', 'SSIM', 'RMSE', 'MAE', 'Hit Rate', 'FAR']
        df_detail = df[column_order].copy()
        # Sort by Scale, Sample, Model
        df_detail = df_detail.sort_values(['Scale', 'Sample', 'Model'])
        df_detail.to_csv(detail_csv_path, index=False, encoding='utf-8-sig')
        print(f"Detailed metrics table for each sample saved to: {detail_csv_path}")

        # 3. Export statistical summary table (mean and standard deviation)
        summary_csv_path = "figs/quantitative_results2.csv"
        # Calculate mean and standard deviation for table display (only calculate numeric columns, exclude Sample column)
        numeric_cols = ['PSNR', 'SSIM', 'RMSE', 'MAE', 'Hit Rate', 'FAR']
        summary = df.groupby(['Scale', 'Model'])[numeric_cols].agg(['mean', 'std'])
        summary.to_csv(summary_csv_path)
        print(f"Data statistics table (mean ± std) saved to: {summary_csv_path}")

        # 4. Plot
        plot_boxplots(df)
    else:
        print("No data read, please check path configuration.")
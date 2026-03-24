# -*- coding: utf-8 -*-
"""
Author :XJQ
Time :2025/12/30 17:00
Function:
"""
# -*- coding: utf-8 -*-
"""
Author : AI Assistant
Time : 2025/12/30
Function: DDS-No-Physics vs DDS-SR ablation experiment comparison box plot
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

# ================= Configuration Region =================
BASE_DIR = "results//normalized_results//"
WATER_THRESHOLD = 0.05

# 1. Only keep two models for ablation experiment comparison
MODEL_MAPPING = {
    "DDS-No-Physics": "DDS-No-Physics",
    "DDS-SR": "DDS-SR"
}

# 2. Modify color scheme: Use contrasting colors (deep blue vs coral red)
PALETTE = [
    "#5DADE2",  # DDS-No-Physics - Light blue
    "#DAA87C",  # DDS-SR         - Coral red
]


# ================= Core Functions (keep consistent) =================

def read_geotiff(path):
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            data = np.nan_to_num(data, nan=0.0)
            data[data < 0] = 0
            return data
    except:
        return None


def calculate_metrics(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))
    d_range = max(np.max(gt), 0.001)

    try:
        val_psnr = psnr(gt, pred, data_range=d_range)
    except:
        val_psnr = 0
    try:
        val_ssim = ssim(gt, pred, data_range=d_range)
    except:
        val_ssim = 0

    pred_mask = pred > WATER_THRESHOLD
    gt_mask = gt > WATER_THRESHOLD
    TP = np.sum(pred_mask & gt_mask)
    FP = np.sum(pred_mask & ~gt_mask)
    FN = np.sum(~pred_mask & gt_mask)
    hit_rate = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0.0

    return {
        "RMSE": rmse, "MAE": mae, "PSNR": val_psnr,
        "SSIM": val_ssim, "Hit Rate": hit_rate, "FAR": far
    }


def collect_data():
    records = []
    scales = ["2m", "4m", "8m"]
    print("Collecting ablation study data...")

    for scale in scales:
        scale_dir = os.path.join(BASE_DIR, f"results_{scale}")
        gt_candidates = glob.glob(os.path.join(scale_dir, "HR*1m"))
        if not gt_candidates: continue
        gt_dir = gt_candidates[0]
        gt_files = [os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.tif"))]

        for display_name, folder_prefix in MODEL_MAPPING.items():
            model_folder_name = f"{folder_prefix}-{scale}"
            model_dir = os.path.join(scale_dir, model_folder_name)

            if not os.path.exists(model_dir):
                print(f"  Skipping {display_name} ({scale}): Folder not found")
                continue

            for fname in gt_files:
                gt_path = os.path.join(gt_dir, fname)
                pred_path = os.path.join(model_dir, fname)
                if os.path.exists(pred_path):
                    gt_data = read_geotiff(gt_path)
                    pred_data = read_geotiff(pred_path)
                    if gt_data is not None and pred_data is not None:
                        metrics = calculate_metrics(pred_data, gt_data)
                        metrics.update({"Scale": scale, "Model": display_name, "Sample": fname})
                        records.append(metrics)
    return pd.DataFrame(records)


def plot_boxplots(df):
    metrics_list = ["PSNR(db)", "SSIM", "RMSE(m)", "MAE(m)", "Hit Rate", "FAR"]
    metrics_map = {"PSNR(db)": "PSNR", "SSIM": "SSIM", "RMSE(m)": "RMSE", "MAE(m)": "MAE", "Hit Rate": "Hit Rate",
                   "FAR": "FAR"}

    sns.set_theme(style="ticks", font="Times New Roman", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    abc = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        # 3. Enable outlier drawing (showfliers=True)
        sns.boxplot(
            data=df, x="Scale", y=metrics_map[metric], hue="Model",
            hue_order=list(MODEL_MAPPING.keys()), ax=ax,
            palette=PALETTE, width=0.6, linewidth=1.5,
            showfliers=True,  # Show outliers
            flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "gray", "alpha": 0.5}
        )

        ax.set_title(f"{abc[i]} {metric}", fontweight='bold', fontsize=15)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontweight='bold', fontsize=16)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=15)

        if ax.get_legend():
            ax.legend(loc='best', fontsize=12, frameon=True)

    plt.tight_layout()
    save_path = "figs/ablation_study_boxplot.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Ablation study chart saved to: {save_path}")


# ================= Main Program =================
if __name__ == "__main__":
    df = collect_data()
    if not df.empty:
        # Save statistical results
        summary = df.groupby(['Scale', 'Model'])[['PSNR', 'SSIM', 'RMSE', 'MAE', 'Hit Rate', 'FAR']].agg(
            ['mean', 'std'])
        summary.to_csv("figs/ablation_quantitative_results.csv")
        print("Statistics table saved to: figs/ablation_quantitative_results.csv")

        # Plot
        plot_boxplots(df)
    else:
        print("No data found.")
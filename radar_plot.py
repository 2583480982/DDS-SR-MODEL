# -*- coding: utf-8 -*-
"""
雷达图绘制脚本
读取 statc.csv，生成一行三列的雷达图，每个子图表示不同分辨率下不同指标的情况
Author: XJQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# ================= 配置区域 =================
# 数据文件路径
DATA_FILE = "statc.csv"

# 分辨率列表（对应3个子图）
# 需与 statc.csv 中的 Scale factor 保持一致
SCALES = ["2m", "4m", "8m"]

# 指标列表（雷达图的轴）
METRICS = ["PSNR", "SSIM", "RMSE", "MAE", "Hit Rate", "FAR"]

# 模型列表（保持与数据一致）
MODELS = ["DDS-SR", "FLO-SR", "UNet-SR", "LR"]

# 专业配色方案（简约大气，新配色）
COLORS = {
    "DDS-SR": "#E3E457",    # 填充颜色
    "FLO-SR": "#A3E3FF",    # 填充颜色
    "LR": "#DAA87C",        # 填充颜色
    "UNet-SR": "#8BCFB5"    # 填充颜色
}

# 边框颜色（比填充颜色更深）
EDGE_COLORS = {
    "DDS-SR": "#B8B045",    # 更深的黄色
    "FLO-SR": "#55AFE2",    # 更深的蓝绿色
    "LR": "#A87A5A",        # 更深的棕色
    "UNet-SR": "#C4BE7A"    # 更深的黄绿色
}

# 归一化范围设置（使用固定范围，比数据范围更大）
NORMALIZATION_RANGES = {
    "PSNR": {"min": 15.0, "max": 35.0},      # PSNR范围
    "SSIM": {"min": 0.70, "max": 1.0},        # SSIM范围
    "RMSE": {"min": 0.0, "max": 0.05},        # RMSE范围
    "MAE": {"min": 0.0, "max": 0.02},         # MAE范围
    "Hit Rate": {"min": 0.70, "max": 1.0},    # Hit Rate范围
    "FAR": {"min": 0.0, "max": 0.25}          # FAR范围
}

# 字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式也使用类似字体


# ================= 核心函数 =================

def normalize_metrics(df, scale):
    """
    归一化指标数据（使用固定范围）
    对于越大越好的指标（PSNR, SSIM, Hit Rate）：归一化到 [0, 1]
    对于越小越好的指标（RMSE, MAE, FAR）：反转后归一化到 [0, 1]
    返回按模型顺序排列的数据字典
    """
    scale_data = df[df['Scale factor'] == scale].copy()
    
    # 按照MODELS列表的顺序排序，确保顺序一致
    scale_data['Model'] = pd.Categorical(scale_data['Model'], categories=MODELS, ordered=True)
    scale_data = scale_data.sort_values('Model')
    
    # 越大越好的指标
    higher_better = ['PSNR', 'SSIM', 'Hit Rate']
    # 越小越好的指标
    lower_better = ['RMSE', 'MAE', 'FAR']
    
    normalized_data = {}
    
    for metric in METRICS:
        values = scale_data[metric].values
        
        # 使用固定范围进行归一化
        range_config = NORMALIZATION_RANGES[metric]
        min_val = range_config['min']
        max_val = range_config['max']
        
        if metric in higher_better:
            # 归一化到 [0, 1]，超出范围的值会被裁剪
            normalized = np.clip((values - min_val) / (max_val - min_val), 0, 1)
        else:  # lower_better
            # 反转后归一化到 [0, 1]，超出范围的值会被裁剪
            normalized = np.clip(1 - (values - min_val) / (max_val - min_val), 0, 1)
        
        normalized_data[metric] = normalized
    
    return normalized_data, scale_data['Model'].values


def plot_radar(ax, data_dict, models, scale, colors, edge_colors):
    """
    在指定轴上绘制雷达图
    """
    # 计算角度（6个指标，均匀分布）
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 设置极坐标
    ax.set_theta_offset(np.pi / 2)  # 从顶部开始
    ax.set_theta_direction(-1)  # 顺时针方向

    # 绘制每个模型（只使用填充面，不用线条和点）
    for i, model in enumerate(models):
        if model not in colors:
            continue

        values = []
        for metric in METRICS:
            if metric in data_dict and i < len(data_dict[metric]):
                values.append(data_dict[metric][i])
            else:
                values.append(0)
        values += values[:1]  # 闭合图形

        # 绘制填充面，并添加加粗边框线条（边框颜色更深）
        fill_color = colors.get(model, '#666666')
        edge_color = edge_colors.get(model, '#333333')  # 使用更深的边框颜色
        ax.fill(angles, values, alpha=0.6, color=fill_color,
                label=model, edgecolor=edge_color, linewidth=2)

    # 设置角度标签（环绕外圆）
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRICS, fontsize=11, fontfamily='Times New Roman')
    ax.tick_params(axis='x', pad=15)

    # 关键：遍历每个标签，设置文字方向平行于圆
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        # 1. 计算标签的旋转角度（matplotlib使用弧度，用度数来设置文本）
        angle_deg = np.rad2deg(angle)

        # 2. 设置旋转角度（核心）：让文字沿角度径向对齐
        label.set_rotation(angle_deg)

        # 3. 微调文字对齐方式（避免文字偏移）
        label.set_ha('center')  # 水平居中
        label.set_va('center')  # 垂直居中

        # 可选：对180°附近的文字额外翻转，避免倒写（比如180°-360°的文字反转180°）
        if 90 < angle_deg < 270:
            label.set_rotation(angle_deg + 180)

    # ========== 修复Y轴径向标签pad生效问题 ==========
    ax.set_ylim(0, 1)
    y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(y_ticks)

    # 手动设置Y轴标签位置，使其更靠近圆心
    y_labels = ['0.2', '0.4', '0.6', '0.8', '1.0']
    ax.set_yticklabels(y_labels, fontsize=9, fontfamily='Times New Roman')

    # 获取Y轴标签并调整其位置
    y_tick_labels = ax.get_yticklabels()
    for i, label in enumerate(y_tick_labels):
        # 通过设置position属性来精确控制标签位置
        # 将标签位置设置得更靠近圆心（减小半径）
        label_radius = y_ticks[i] - 0.05  # 向内偏移0.05个单位
        # 注意：这种方法在matplotlib中比较复杂，我们主要通过pad参数控制

    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='gray')

    # 关键1：先重置rlabel的变换（取消角度固定对pad的覆盖）
    ax.tick_params(axis='y',  pad=10)  # 进一步增加负值使标签更靠近圆心
    # 关键2：重新设置rlabel位置（需在tick_params之后）
    ax.set_rlabel_position(90)
    ax.yaxis.set_label_position('left')

    # 可选：进一步调整径向标签的位置
    # 获取当前的径向标签并手动调整它们的位置
    # 获取当前的径向标签并手动调整它们的位置
    for label in ax.get_yticklabels():
        pos = label.get_position()
        # 解析半径并减小
        if len(pos) == 2:
            r = np.hypot(pos[0], pos[1])
        else:
            r = pos[1]
        r -= 1  # 靠近圆心
        # 固定90°角度，重新计算坐标
        theta = np.radians(90)
        new_x = r * np.cos(theta)
        new_y = r * np.sin(theta)
        label.set_position((new_x, new_y))
        label.set_horizontalalignment('right')  # 保持文字对齐

    # # 设置Y轴标签的背景透明，避免遮挡网格线
    # for label in ax.get_yticklabels():
    #     label.set_backgroundcolor('white')
    #     label.set_alpha(0.8)

    # 添加一个小圆点在中心，帮助视觉定位
    ax.plot(0, 0, 'o', markersize=3, color='black', zorder=10)


    # 加粗最外圈的实线圆（图框）
    # 在极坐标图中，最外层圆通过spine控制
    ax.spines['polar'].set_linewidth(2)
    ax.spines['polar'].set_color('black')
    # 绘制最外层粗实线圆（在数据之后绘制，确保在最上层）
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.ones_like(theta)
    ax.plot(theta, r, 'k-', linewidth=2.5, zorder=100)

    # 添加标题
    ax.set_title(f'Scale factor:{scale} ', fontsize=14, fontweight='bold', pad=20, fontfamily='Times New Roman')


def main():
    """主函数"""
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv(DATA_FILE)
    
    # 清理数据（去除空格）
    df['Model'] = df['Model'].str.strip()
    df['Scale factor'] = df['Scale factor'].str.strip()
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    
    # 创建图形：一行三列
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    # 为每个分辨率绘制雷达图
    for idx, scale in enumerate(SCALES):
        print(f"正在处理 {scale} 分辨率...")
        
        # 归一化数据
        normalized_data, models = normalize_metrics(df, scale)
        
        # 绘制雷达图
        plot_radar(axes[idx], normalized_data, models, scale, COLORS, EDGE_COLORS)
    
    # 添加统一图例（放在底部）
    # 由于不再使用线条，需要从填充面获取图例
    from matplotlib.patches import Patch
    handles = []
    labels = []
    for model in MODELS:
        if model in COLORS:
            # 创建一个临时填充面用于图例（边框颜色更深）
            handles.append(Patch(facecolor=COLORS[model], alpha=0.4, edgecolor=EDGE_COLORS.get(model, '#333333')))
            labels.append(model)
    
    fig.legend(handles, labels, loc='lower center', ncol=len(MODELS), 
               bbox_to_anchor=(0.5, -0.005), frameon=True, fontsize=12,
               fancybox=True, shadow=False, prop={'family': 'Times New Roman'})
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 保存图片
    save_path = "radar_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"雷达图已保存至: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()


# DDS-SR: 地形引导的双流物理约束水深超分辨率模型

## 项目简介

本研究提出一种地形引导的双流物理约束重建框架，用于将低分辨率水深模拟结果高效增强为高分辨率产品。该框架采用并行特征路径分别编码低分辨率水深与高分辨率地形信息，并引入跨模态空间注意力以强化地形控制区域对重建过程的调制作用；同时通过坡度一致性、水面连续性与局部平滑性等物理软约束，引导输出结果向更合理的水力学可行域收敛。

## 主要特性

- **双流架构**：并行处理低分辨率水深和高分辨率DEM地形信息
- **跨模态空间注意力**：强化地形对重建过程的调制作用
- **物理约束损失**：
  - 坡度一致性约束
  - 水面连续性约束
  - 局部平滑性约束
- **高效增强**：将4m/8m分辨率水深数据增强至1m高分辨率

## 模型架构

### 核心组件

1. **双流编码器**
   - 水深流：处理低分辨率水深数据
   - 地形流：处理高分辨率DEM数据

2. **跨模态空间注意力模块**
   - 融合水深和地形特征
   - 强化地形控制区域的影响

3. **物理约束损失函数**
   - 重建损失（MAE）
   - 坡度一致性损失
   - 水面连续性损失
   - 局部平滑性损失

## 文件结构

```
DDS_SR_MODEL/
├── DDS-SR.py                    # DDS-SR主模型文件
├── FLO_SR.py                    # FLO-SR对比模型
├── UNet_SR.py                   # UNet对比模型
├── SR_gan.py                    # SRGAN对比模型
├── apply_DDS-SR.py              # DDS-SR推理脚本
├── apply_FLO_SR.py              # FLO-SR推理脚本
├── apply_UNet_model.py          # UNet推理脚本
├── PLO_box.py                  # 模型对比可视化
├── plot_box2.py                # 消融实验可视化
├── raster_normalization.py     # 标准化栅格数据
└── DDS-SR-modularization/      # 模块化代码
    ├── config.py                # 配置文件
    ├── model.py                # 模型定义
    ├── train.py                # 训练脚本
    ├── losses.py               # 损失函数
    ├── data_loader.py          # 数据加载器
    ├── trainer.py             # 训练器
    ├── inference.py           # 推理脚本
    └── utils.py              # 工具函数
```

## 环境要求

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Rasterio
- scikit-image
......
参考environment.yml

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/DDS-SR.git
cd DDS-SR

# 创建并按照环境
conda env create -f environment.yml
```

## 数据准备

训练数据需要以下文件夹：

```
date/
├── Label_1m_train/          # 训练集高分辨率水深（1m）
├── Feature_4m_train/        # 训练集低分辨率水深（4m）
├── DEM-01m/                # 高分辨率DEM（1m）
├── Label_1m_val/            # 验证集高分辨率水深
└── Feature_4m_val/          # 验证集低分辨率水深
```

## 使用方法

### 训练模型

```bash
# 使用模块化代码训练
cd DDS-SR-modularization
python train.py --config config.py
```

### 模型推理

```bash
# DDS-SR推理
python apply_DDS-SR.py --input_path input_folder --output_path output_folder

# FLO-SR推理
python apply_FLO_SR.py --input_path input_folder --output_path output_folder

# UNet推理
python apply_UNet_model.py --input_path input_folder --output_path output_folder
```
### 标准化栅格数据

```bash
# 标准化栅格数据，将缩放0-255的水深值标准化到真实水深之间
python raster_normalization.py
```

### 结果可视化

```bash
# 模型对比箱型图
python PLO_box.py

# 消融实验对比
python plot_box2.py
```
## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@article{your_paper_2025,
  title={地形引导的双流物理约束水深超分辨率重建},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- Email: jq.xiao@hhu.edu.cn
- GitHub Issues: https://github.com/yourusername/DDS-SR/issues

## 致谢

感谢所有为本项目做出贡献的研究者和开发者。

## 更新日志

### v1.0.0 (2025-12-30)
- 初始版本发布
- 实现DDS-SR核心模型
- 提供训练和推理脚本
- 添加对比模型（FLO-SR、UNet-SR、SRGAN）
- 实现消融实验功能

---

**注意**：本代码仅用于学术研究目的。如需商业使用，请联系作者获取授权。

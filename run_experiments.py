"""
消融实验批处理脚本
依次运行多个消融实验模式的训练和推理
"""

import os
import subprocess
import time
import sys

def run_command(command):
    """运行Shell命令"""
    print(f"\n执行命令: {command}")
    print("="*60)
    
    # 使用subprocess.run执行命令，并实时输出
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=sys.stdout, 
        stderr=sys.stderr
    )
    process.communicate()
    
    if process.returncode != 0:
        print(f"命令执行失败，退出码: {process.returncode}")
        return False
    return True

def main():
    # 定义要运行的实验模式
    experiments = [
        # 模式名称, 描述
        # ('baseline_single_stream', 'Baseline (单流/无注意力/无物理)'),
        # ('simple_concat', 'Simple Concat (双流/简单拼接/无物理)'),
        ('no_physics', 'No Physics (双流/跨模态注意力/无物理)'),
        ('no_attention', 'No Attention (双流/简单拼接/有物理)')
    ]
    
    # 训练设置
    epochs = 200  # 可以根据需要调整
    
    start_time_total = time.time()
    
    print("开始运行消融实验组...")
    print(f"共 {len(experiments)} 个实验")
    
    for i, (mode, desc) in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"实验 {i}/{len(experiments)}: {mode}")
        print(f"描述: {desc}")
        print(f"{'#'*80}")
        
        # 1. 训练
        print(f"\n>>> 开始训练 [{mode}]...")
        train_cmd = f"python train.py --ablation_mode {mode} --epochs {epochs}"
        if not run_command(train_cmd):
            print(f"实验 {mode} 训练失败，跳过后续步骤...")
            continue
            
        # 2. 推理
        print(f"\n>>> 开始推理 [{mode}]...")
        # 构造路径（与train.py中的自动命名逻辑一致）
        model_path = f"out_save/DDS_SR_best_4x_{mode}.h5"
        output_folder = f"DDS_SR_test_best_4x_{mode}/"
        
        inference_cmd = (
            f"python inference.py "
            f"--model {model_path} "
            f"--mode batch "
            f"--output {output_folder} "
            f"--visualize"
        )
        
        if not run_command(inference_cmd):
            print(f"实验 {mode} 推理失败...")
        
        print(f"\n✓ 实验 {mode} 完成")
        
    total_time = time.time() - start_time_total
    print(f"\n\n{'='*80}")
    print(f"所有实验完成！总耗时: {total_time/3600:.2f} 小时")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()


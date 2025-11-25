#!/usr/bin/env python3
"""
实验结果可视化工具
生成三个主要图表：
1. 隐私-效用权衡曲线
2. 消融实验柱状图
3. Rank敏感度折线图
"""

import argparse
import pickle
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 尝试导入外部配置
try:
    from run_main import EXPERIMENT_CONFIGS
except ImportError:
    print("❌ 错误: 无法导入 'run_main.py'。请确保该文件在当前目录下或 PYTHONPATH 中。")
    sys.exit(1)

# ========== 全局配置 ==========
DEFAULT_OUTPUT_DIR = Path.home() / 'data/sepfpl/p_outputs'
DEFAULT_TAIL_EPOCHS = 3
DEFAULT_FIG_DIR = Path('figures')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


# ========== 数据读取函数（复用 table.py 的逻辑） ==========

def tail_values(values: List[float], tail: int = DEFAULT_TAIL_EPOCHS) -> List[float]:
    """获取列表末尾的 N 个值"""
    if not values:
        return []
    if tail is None or len(values) <= tail:
        return list(values)
    return list(values[-tail:])


def load_metrics(file_path: Path) -> Tuple[List[float], List[float]]:
    """从 pickle 文件加载数据"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return [], []

    local_hist, neighbor_hist = [], []

    if isinstance(data, dict):
        local_hist = data.get('local_acc', [])
        neighbor_hist = data.get('neighbor_acc', [])
    elif isinstance(data, (list, tuple)):
        if len(data) >= 2:
            local_hist = data[0] if isinstance(data[0], list) else []
            neighbor_hist = data[1] if isinstance(data[1], list) else []

    return local_hist or [], neighbor_hist or []


def find_output_file(base_dir: Path, pattern_base: str) -> Optional[Path]:
    """查找文件"""
    possible_names = [f'{pattern_base}.pkl', f'{pattern_base}_10.pkl']
    for name in possible_names:
        file_path = base_dir / name
        if file_path.exists():
            return file_path
    return None


def read_accuracy(exp_name: str, dataset: str, factorization: str, rank: int,
                  noise: float, seed_list: List[int], num_users: Optional[int],
                  output_base_dir: Path, tail_epochs: int, use_neighbor: bool = False) -> Tuple[float, float]:
    """
    读取准确率数据，返回均值和标准差
    
    Args:
        use_neighbor: 如果 True，返回 neighbor accuracy；否则返回 local accuracy
    """
    per_seed_values = []
    base_dir = output_base_dir / exp_name / dataset

    for seed in seed_list:
        if num_users is not None:
            pattern = f'acc_{factorization}_{rank}_{noise}_{seed}_{num_users}'
        else:
            pattern = f'acc_{factorization}_{rank}_{noise}_{seed}'
        
        file_path = find_output_file(base_dir, pattern)
        if not file_path:
            continue
        
        local_hist, neighbor_hist = load_metrics(file_path)
        hist = neighbor_hist if use_neighbor else local_hist
        if hist:
            per_seed_values.extend(tail_values(hist, tail_epochs))
    
    if not per_seed_values:
        return 0.0, 0.0
    
    avg = mean(per_seed_values)
    std = stdev(per_seed_values) if len(per_seed_values) > 1 else 0.0
    return avg, std


# ========== 图1: 隐私-效用权衡曲线 ==========

def plot_privacy_utility_tradeoff(output_dir: Path = DEFAULT_OUTPUT_DIR, 
                                   tail_epochs: int = DEFAULT_TAIL_EPOCHS,
                                   fig_dir: Path = DEFAULT_FIG_DIR,
                                   use_neighbor: bool = False):
    """
    绘制隐私-效用权衡曲线
    
    X轴: 隐私预算 ε (Noise level)
    Y轴: 测试准确率
    每个数据集一张子图
    """
    config = EXPERIMENT_CONFIGS['EXPERIMENT_1_SIMPLE']
    exp_name = config['exp_name']
    datasets = config['dataset_list']
    methods = config['factorization_list']
    noise_list = config['noise_list']
    rank = config['rank_list'][0]
    seed_list = config['seed_list']
    num_users = config['num_users_list'][0]
    
    # 方法名称映射（用于图例）
    method_labels = {
        'promptfl': 'PromptFL',
        'fedotp': 'FedOTP',
        'fedpgp': 'FedPGP',
        'dpfpl': 'DP-FPL',
        'sepfpl': 'SepFPL'
    }
    
    # 颜色映射
    colors = {
        'promptfl': '#1f77b4',
        'fedotp': '#ff7f0e',
        'fedpgp': '#2ca02c',
        'dpfpl': '#d62728',
        'sepfpl': '#9467bd'
    }
    
    # 创建子图
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Privacy-Utility Tradeoff', fontsize=16, fontweight='bold', y=1.02)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        for method in methods:
            accuracies = []
            stds = []
            valid_noises = []
            
            for noise in noise_list:
                acc, std = read_accuracy(exp_name, dataset, method, rank, noise,
                                        seed_list, num_users, output_dir, tail_epochs, use_neighbor)
                if acc > 0:  # 只绘制有效数据
                    accuracies.append(acc)
                    stds.append(std)
                    valid_noises.append(noise)
            
            if accuracies:
                label = method_labels.get(method, method)
                color = colors.get(method, '#000000')
                ax.plot(valid_noises, accuracies, marker='o', label=label, 
                       color=color, linewidth=2, markersize=6)
                ax.errorbar(valid_noises, accuracies, yerr=stds, 
                           color=color, alpha=0.3, capsize=3)
        
        ax.set_xlabel('Privacy Budget ε (Noise Level)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(dataset.replace('-', ' ').title(), fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(-0.02, max(noise_list) * 1.1)
    
    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / 'privacy_utility_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图1已保存: {output_path}")
    plt.close()


# ========== 图2: 消融实验柱状图 ==========

def plot_ablation_study(output_dir: Path = DEFAULT_OUTPUT_DIR,
                        tail_epochs: int = DEFAULT_TAIL_EPOCHS,
                        fig_dir: Path = DEFAULT_FIG_DIR,
                        use_neighbor: bool = False):
    """
    绘制消融实验柱状图
    
    X轴: 噪声水平
    组内柱子: DP-FPL (Base), +HCSE, +TimeAdaptive, SepFPL (Full)
    """
    config = EXPERIMENT_CONFIGS['EXPERIMENT_2_ABLATION']
    exp_name = config['exp_name']
    datasets = config['dataset_list']
    methods = ['dpfpl', 'sepfpl_hcse', 'sepfpl_time_adaptive', 'sepfpl']
    method_labels = {
        'dpfpl': 'DP-FPL\n(Base)',
        'sepfpl_hcse': 'DP-FPL\n+HCSE',
        'sepfpl_time_adaptive': 'DP-FPL\n+TimeAdaptive',
        'sepfpl': 'SepFPL\n(Full)'
    }
    noise_list = config['noise_list']
    rank = 8  # 使用固定的 rank=8
    seed_list = config['seed_list']
    num_users = config['num_users_list'][0]
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Ablation Study: Component Contribution', fontsize=16, fontweight='bold', y=1.02)
    
    x = np.arange(len(noise_list))
    width = 0.2  # 柱子宽度
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        for i, method in enumerate(methods):
            accuracies = []
            stds = []
            
            for noise in noise_list:
                acc, std = read_accuracy(exp_name, dataset, method, rank, noise,
                                        seed_list, num_users, output_dir, tail_epochs, use_neighbor)
                accuracies.append(acc)
                stds.append(std)
            
            offset = (i - len(methods) / 2 + 0.5) * width
            bars = ax.bar(x + offset, accuracies, width, label=method_labels[method],
                         color=colors_list[i], alpha=0.8, yerr=stds, capsize=3)
        
        ax.set_xlabel('Noise Level (ε)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(dataset.replace('-', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{n:.2f}' for n in noise_list])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / 'ablation_study.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图2已保存: {output_path}")
    plt.close()


# ========== 图3: Rank敏感度折线图 ==========

def plot_rank_sensitivity(output_dir: Path = DEFAULT_OUTPUT_DIR,
                          tail_epochs: int = DEFAULT_TAIL_EPOCHS,
                          fig_dir: Path = DEFAULT_FIG_DIR,
                          use_neighbor: bool = False):
    """
    绘制Rank敏感度折线图
    
    X轴: Rank r ∈ {1, 2, 4, 8, 16}
    Y轴: 准确率
    """
    config = EXPERIMENT_CONFIGS['EXPERIMENT_2_ABLATION']
    exp_name = config['exp_name']
    datasets = config['dataset_list']
    methods = ['dpfpl', 'sepfpl']
    method_labels = {
        'dpfpl': 'DP-FPL',
        'sepfpl': 'SepFPL'
    }
    rank_list = config['rank_list']
    noise_list = config['noise_list']
    seed_list = config['seed_list']
    num_users = config['num_users_list'][0]
    
    # 为每个数据集和每个噪声水平创建子图
    n_datasets = len(datasets)
    n_noises = len(noise_list)
    fig, axes = plt.subplots(n_datasets, n_noises, figsize=(5 * n_noises, 4 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    if n_noises == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Rank Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    colors_list = ['#d62728', '#9467bd']
    
    for d_idx, dataset in enumerate(datasets):
        for n_idx, noise in enumerate(noise_list):
            ax = axes[d_idx, n_idx]
            
            for m_idx, method in enumerate(methods):
                accuracies = []
                stds = []
                valid_ranks = []
                
                for rank in rank_list:
                    acc, std = read_accuracy(exp_name, dataset, method, rank, noise,
                                            seed_list, num_users, output_dir, tail_epochs, use_neighbor)
                    if acc > 0:
                        accuracies.append(acc)
                        stds.append(std)
                        valid_ranks.append(rank)
                
                if accuracies:
                    label = method_labels[method]
                    ax.plot(valid_ranks, accuracies, marker='o', label=label,
                           color=colors_list[m_idx], linewidth=2, markersize=6)
                    ax.errorbar(valid_ranks, accuracies, yerr=stds,
                               color=colors_list[m_idx], alpha=0.3, capsize=3)
            
            ax.set_xlabel('Rank (r)', fontsize=11)
            if n_idx == 0:
                ax.set_ylabel('Test Accuracy (%)', fontsize=11)
            ax.set_title(f'{dataset.replace("-", " ").title()}\nε={noise}', fontsize=11, fontweight='bold')
            ax.set_xticks(rank_list)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / 'rank_sensitivity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图3已保存: {output_path}")
    plt.close()


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验结果可视化工具")
    
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="数据目录")
    parser.add_argument("--tail-epochs", type=int, default=DEFAULT_TAIL_EPOCHS, help="统计轮次")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help="图片保存目录")
    parser.add_argument("--use-neighbor", action="store_true", help="使用 neighbor accuracy 而非 local accuracy")
    
    parser.add_argument("--plot1", action="store_true", help="绘制图1: 隐私-效用权衡曲线")
    parser.add_argument("--plot2", action="store_true", help="绘制图2: 消融实验柱状图")
    parser.add_argument("--plot3", action="store_true", help="绘制图3: Rank敏感度折线图")
    parser.add_argument("--all", action="store_true", help="绘制所有图表")
    
    args = parser.parse_args()
    
    if not (args.plot1 or args.plot2 or args.plot3 or args.all):
        print("⚠️  未指定要绘制的图表，使用 --all 绘制所有图表，或使用 --plot1/--plot2/--plot3 选择特定图表")
        args.all = True
    
    if args.all or args.plot1:
        print("\n📊 正在绘制图1: 隐私-效用权衡曲线...")
        plot_privacy_utility_tradeoff(args.output_dir, args.tail_epochs, args.fig_dir, args.use_neighbor)
    
    if args.all or args.plot2:
        print("\n📊 正在绘制图2: 消融实验柱状图...")
        plot_ablation_study(args.output_dir, args.tail_epochs, args.fig_dir, args.use_neighbor)
    
    if args.all or args.plot3:
        print("\n📊 正在绘制图3: Rank敏感度折线图...")
        plot_rank_sensitivity(args.output_dir, args.tail_epochs, args.fig_dir, args.use_neighbor)
    
    print(f"\n✅ 所有图表已保存到: {args.fig_dir}")


if __name__ == "__main__":
    main()


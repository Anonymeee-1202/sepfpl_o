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

# 导入table.py中的后处理函数
try:
    from table import postprocess_results, extract_value, read_data, read_scheme
except ImportError:
    print("❌ 错误: 无法导入 'table.py'。请确保该文件在当前目录下或 PYTHONPATH 中。")
    sys.exit(1)

# ========== 全局配置 ==========
# 注意：默认使用outputs目录（与table.py一致），如果数据在p_outputs，请使用--output-dir参数指定
DEFAULT_OUTPUT_DIR = Path.home() / 'data/sepfpl/outputs'
DEFAULT_TAIL_EPOCHS = 10  # 与table.py保持一致
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


# ========== 图4（新增）: Exp1 噪声折线图 ==========

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple

# 假设 matplotlib 和 numpy 已经导入
# 建议在文件头部导入 seaborn，如果没有安装，可以用 matplotlib 原生实现，
# 但下面的代码尽量只用 matplotlib 以减少依赖，同时模拟 seaborn 的美观度。

def _parse_stat_value(stat_str: str) -> Tuple[float, float]:
    """将 '85.20 ± 1.05' 解析为 (85.20, 1.05)。"""
    if not stat_str or stat_str == "N/A":
        return 0.0, 0.0
    try:
        parts = stat_str.split('±')
        mean_val = float(parts[0].strip())
        std_val = float(parts[1].strip()) if len(parts) > 1 else 0.0
        return mean_val, std_val
    except (ValueError, IndexError):
        return 0.0, 0.0

def plot_exp1_noise_linecharts(output_dir: Path = DEFAULT_OUTPUT_DIR,
                               tail_epochs: int = DEFAULT_TAIL_EPOCHS,
                               fig_dir: Path = DEFAULT_FIG_DIR):
    """
    根据 Exp1 (Simple + Hard) 的结果，绘制符合学术发表标准的噪声-准确率折线图。
    """
    
    # --- 1. 学术绘图风格设置 ---
    # 使用类似 LaTeX 的字体渲染，增强专业感
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],  # 论文常用字体
        'mathtext.fontset': 'stix',         # 数学公式字体类似 LaTeX
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.linewidth': 1.2,              # 坐标轴线加粗
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,             # 线条加粗
        'lines.markersize': 9,              # 标记点变大
    })

    target_configs = [
        ('EXPERIMENT_1_SIMPLE', 'exp1_simple'),
        ('EXPERIMENT_1_HARD', 'exp1_hard'),
    ]

    method_labels = {
        'promptfl': 'PromptFL',
        'fedotp': 'FedOTP',
        'fedpgp': 'FedPGP',
        'dpfpl': 'DP-FPL',
        'sepfpl': 'SepFPL'
    }
    
    method_colors = {
        'promptfl': '#1f77b4', # Blue
        'fedotp': '#ff7f0e',   # Orange
        'fedpgp': '#2ca02c',   # Green
        'dpfpl': '#d62728',    # Red
        'sepfpl': '#9467bd'    # Purple
    }

    # --- 2. 增加 Marker 映射 ---
    # 黑白打印时，仅靠颜色很难区分，必须加不同的 marker
    method_markers = {
        'promptfl': 'o',  # Circle
        'fedotp': 's',    # Square
        'fedpgp': '^',    # Triangle Up
        'dpfpl': 'D',     # Diamond
        'sepfpl': 'X'     # X (thick)
    }

    for config_key, file_suffix in target_configs:
        if config_key not in EXPERIMENT_CONFIGS:
            continue

        config = EXPERIMENT_CONFIGS[config_key]
        exp_name = config['exp_name']
        datasets = config['dataset_list']
        methods = config['factorization_list']
        noise_list = config['noise_list']
        rank = config['rank_list'][0] if config.get('rank_list') else config.get('rank', 8)
        seed_list = config['seed_list']
        num_users_list = config.get('num_users_list') or [config.get('num_users', 10)]

        dataset_entries = []
        for dataset in datasets:
            for num_users in num_users_list:
                dataset_entries.append((dataset, num_users))

        if not dataset_entries:
            continue

        metric_set = [
            ('Local Accuracy (%)', False, 'local'),
            ('Neighbor Accuracy (%)', True, 'neighbor')
        ]

        if config_key == 'EXPERIMENT_1_SIMPLE':
            n_rows, n_cols = 2, 2
        else:
            n_panels = len(dataset_entries)
            n_cols = min(3, n_panels)
            n_rows = (n_panels + n_cols - 1) // n_cols

        x_positions = np.arange(len(noise_list))
        # 优化 tick labels 显示
        x_tick_labels = ['none'] + [f'{n}' for n in noise_list[1:]] 
        exp_type = 'exp1'

        for metric_label, metric_neighbor, metric_suffix in metric_set:
            # 调整 figure size，使其更饱满
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
            axes = axes.flatten()

            for idx, (dataset, num_users) in enumerate(dataset_entries):
                ax = axes[idx]
                method_acc_map = {m: {'acc': [], 'std': []} for m in methods}

                # --- 数据读取逻辑保持不变 ---
                for noise in noise_list:
                    l_list, n_list = read_scheme(
                        exp_name, dataset, rank, noise, methods,
                        seed_list, num_users, output_dir, tail_epochs
                    )

                    processed_list = postprocess_results(
                        n_list if metric_neighbor else l_list,
                        methods,
                        exp_type
                    )

                    for m_idx, method in enumerate(methods):
                        stat_str = processed_list[m_idx] if m_idx < len(processed_list) else "N/A"
                        mean_val, std_val = _parse_stat_value(stat_str)
                        method_acc_map[method]['acc'].append(mean_val)
                        method_acc_map[method]['std'].append(std_val)
                
                # --- 绘图核心优化 ---
                for method in methods:
                    accuracies = method_acc_map[method]['acc']
                    stds = method_acc_map[method]['std']

                    if any(acc > 0 for acc in accuracies):
                        label = method_labels.get(method, method)
                        color = method_colors.get(method, '#333333')
                        marker = method_markers.get(method, 'o')
                        
                        # 绘制主线
                        ax.plot(x_positions, accuracies, 
                                marker=marker, 
                                label=label,
                                color=color, 
                                linewidth=2.5, 
                                markersize=8,
                                markeredgecolor='white', # 标记边缘白色，增加对比度
                                markeredgewidth=1.5,
                                zorder=10) # 保证线在网格之上

                        # 绘制误差带
                        ax.fill_between(
                            x_positions,
                            [a - s for a, s in zip(accuracies, stds)],
                            [a + s for a, s in zip(accuracies, stds)],
                            color=color, 
                            alpha=0.15, 
                            edgecolor=None, # 去掉误差带边框
                            zorder=5
                        )

                # --- 标题和轴标签优化 ---
                title = dataset.replace('-', ' ').title()
                if 'Cifar' in title: title = title.replace('Cifar', 'CIFAR') # 特殊大小写修正
                
                # 如果用户数不同才显示用户数，否则标题太长
                if len(num_users_list) > 1:
                    title += f' ($N={num_users}$)'
                
                ax.set_title(title, fontweight='bold', pad=12)

                # 仅在第一列显示 Y 轴标签
                if idx % n_cols == 0:
                    ax.set_ylabel(metric_label, fontweight='bold')
                
                # 仅在最后一行显示 X 轴标签 (为了紧凑布局，可选)
                # if idx >= (n_rows - 1) * n_cols: 
                ax.set_xlabel(r'Noise Level $\epsilon$', fontweight='bold')
                
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_tick_labels)
                
                # --- 网格和边框美化 (Academic Style) ---
                ax.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=0)
                
                # 移除右边和上边的边框 (Despine)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # 设定 Y 轴下限，留出一点 buffer
                # 自动计算下限可能更好，这里保留你的逻辑但加点 buffer
                all_accs = [val for m in methods for val in method_acc_map[m]['acc'] if val > 0]
                if all_accs:
                    min_acc = min(all_accs)
                    ax.set_ylim(bottom=max(0, min_acc - 5), top=102) # 上限稍微超过100一点点以免截断误差带

            # 隐藏多余的子图
            for empty_idx in range(len(dataset_entries), len(axes)):
                axes[empty_idx].set_visible(False)

            # --- 图例优化 ---
            handles, labels = axes[0].get_legend_handles_labels()
            # 将图例放在图的顶部外侧，横向排列，且去掉边框
            fig.legend(handles, labels, 
                       loc='lower center', 
                       bbox_to_anchor=(0.5, 1.0), # 放在整个图的上方
                       ncol=len(methods), 
                       frameon=False, # 去掉图例边框
                       columnspacing=1.5,
                       handletextpad=0.4)

            plt.tight_layout()
            # 预留顶部空间给 Legend
            # plt.subplots_adjust(top=0.90) 
            
            fig_dir.mkdir(parents=True, exist_ok=True)
            suffix = metric_suffix
            output_path = fig_dir / f'exp1_noise_curve_{file_suffix}_{suffix}.pdf' # 推荐保存为 PDF
            
            # 同时保存 PNG 和 PDF。PDF 用于论文插入（矢量图），PNG 用于预览
            plt.savefig(output_path, bbox_inches='tight')
            # plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
            
            print(f"✅ Exp1 噪声折线图已保存: {output_path}")
            plt.close()

# 恢复默认 RC 参数以防影响后续代码（可选）
# plt.rcParams.update(plt.rcParamsDefault)


def plot_exp2_bar_charts(output_dir: Path = DEFAULT_OUTPUT_DIR,
                         tail_epochs: int = DEFAULT_TAIL_EPOCHS,
                         fig_dir: Path = DEFAULT_FIG_DIR,
                         use_postprocess: bool = True):
    """
    绘制exp2的柱状图 (学术论文风格优化版)
    
    改进点:
    - Times New Roman 字体
    - 更加专业的配色 (Colorblind-friendly / Academic)
    - 移除 Top/Right Spines
    - 添加柱状图边框
    - 优化网格线层级
    """
    
    # --- 全局绘图风格设置 ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',  # 数学公式字体与Times兼容
        'font.size': 14 + 4,
        'axes.labelsize': 18 + 4,
        'axes.titlesize': 18 + 8,
        'xtick.labelsize': 16 + 4,
        'ytick.labelsize': 16 + 4,
        'legend.fontsize': 16 + 4,
        'axes.linewidth': 1.2, # 坐标轴线宽
    })

    config = EXPERIMENT_CONFIGS['EXPERIMENT_2_ABLATION']
    exp_name = config['exp_name']
    datasets = config['dataset_list']
    methods = config['factorization_list']
    rank_list = config['rank_list']
    noise_list = config['noise_list']
    seed_list = config['seed_list']
    num_users = config['num_users_list'][0]
    exp_type = 'exp2'
    
    # 方法名称映射
    method_labels = {
        'dpfpl': 'w/o TimeAdaptive & SE',
        'sepfpl_time_adaptive': 'w/ TimeAdaptive',
        'sepfpl_hcse': 'w/ SE',
        'sepfpl': '(SepFPL) Full Method' 
    }
    
    # 学术风格配色 (Muted/Deep Colors)
    # 对应 noise: 0.4 (High), 0.1 (Mid), 0.01 (Low)
    noise_colors = {
        0.4: '#4E79A7',   # 偏灰蓝
        0.1: '#F28E2B',   # 偏柔和橙
        0.01: '#59A14F'   # 偏深绿
    }
    
    # 图例标签映射
    noise_labels = {
        0.4: r'$\epsilon=0.4$',
        0.1: r'$\epsilon=0.1$',
        0.01: r'$\epsilon=0.01$'
    }

    for dataset in datasets:
        # 动态调整Y轴下限，保留更多视觉空间
        y_min = 70 if dataset == 'caltech-101' else 50
        y_max = 95 if dataset == 'caltech-101' else 85

        for use_neighbor in [False, True]:
            acc_type = 'neighbor' if use_neighbor else 'local'
            
            n_methods = len(methods)
            # 增加高度以容纳底部标签，增加宽度防止拥挤
            fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), sharey=True)
            if n_methods == 1:
                axes = [axes]
            
            x_pos = np.arange(len(rank_list))
            width = 0.25  # 稍微调窄一点，增加间隙感
            
            # 遍历每个方法绘制子图
            for m_idx, method in enumerate(methods):
                ax = axes[m_idx]
                
                # 网格线置于底层 (zorder=0)
                ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)
                
                for n_idx, noise in enumerate(noise_list):
                    accuracies = []
                    stds = []
                    
                    for rank in rank_list:
                        # 数据读取逻辑保持不变
                        try:
                            l_list, n_list = read_scheme(
                                exp_name, dataset, rank, noise, methods,
                                seed_list, num_users, output_dir, tail_epochs
                            )
                            
                            if use_postprocess:
                                l_proc = postprocess_results(l_list, methods, exp_type)
                                n_proc = postprocess_results(n_list, methods, exp_type)
                            else:
                                l_proc = l_list
                                n_proc = n_list
                            
                            stat_list = n_proc if use_neighbor else l_proc
                            method_idx = methods.index(method)
                            stat_str = stat_list[method_idx] if method_idx < len(stat_list) else "N/A"
                            
                            if stat_str and stat_str != "N/A":
                                parts = stat_str.split('±')
                                mean_val = float(parts[0].strip())
                                std_val = float(parts[1].strip()) if len(parts) > 1 else 0.0
                                accuracies.append(mean_val)
                                stds.append(std_val)
                            else:
                                accuracies.append(0.0)
                                stds.append(0.0)
                        except Exception as e:
                            print(f"Error reading data for {method}, rank {rank}, noise {noise}: {e}")
                            accuracies.append(0.0)
                            stds.append(0.0)
                    
                    # 绘制柱状图
                    offset = (n_idx - 1) * width
                    # zorder=3 确保柱子在网格线之上
                    # edgecolor='black', linewidth=0.8 增加边缘清晰度
                    ax.bar(x_pos + offset, accuracies, width, 
                           label=noise_labels[noise],
                           color=noise_colors[noise], 
                           edgecolor='black',
                           linewidth=0.8,
                           alpha=0.9,
                           zorder=3)

                # --- 子图美化 ---
                
                # 移除顶部和右侧边框 (Despine)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # X轴标签简化
                ax.set_xticks(x_pos)
                # 将 '16' 替换为 'Full' 或者保持数字，视论文语境而定，这里保持简洁
                x_labels = [str(r) if r != 16 else 'Full' for r in rank_list]
                ax.set_xticklabels(x_labels)
                
                # 标题处理
                title_label = method_labels.get(method, method)
                ax.set_title(title_label, pad=15, fontsize=22, fontweight='bold')

                # Y轴处理
                ax.set_ylim(bottom=y_min, top=y_max)
                ax.set_yticks(np.arange(y_min, y_max + 1, 5))
                if m_idx == 0:
                    ylabel_text = 'Neighbor Accuracy (%)' if use_neighbor else 'Local Accuracy (%)'
                    ax.set_ylabel(ylabel_text, fontweight='bold', labelpad=10)
                
                # 为每个子图添加下方 Rank 标签 (或者在整图添加，这里选择每个子图添加更清晰)
                ax.set_xlabel(r'Rank ($r$)', fontsize=16)

            # --- 整体图例与布局 ---
            
            # 获取句柄和标签 (从第一个子图)
            handles, labels = axes[0].get_legend_handles_labels()
            
            # 图例放在底部居中，水平排列 (Paper常用布局)
            # 或者放在右侧 (User原意)，这里优化右侧布局
            fig.legend(handles, labels, 
                      loc='center right',
                      bbox_to_anchor=(0.99, 0.5),
                      frameon=False, # 去掉图例边框，更简洁
                      fontsize=16,
                      title="Noise Level",
                      title_fontsize=16,
                      handlelength=1.5,  # 减少图例项长度
                      handletextpad=0.3,  # 减少文本与标记的间距
                      columnspacing=0.8)  # 减少列间距
            
            # 调整布局
            plt.tight_layout()
            # 再次手动调整边距，减小子图间距并为右侧图例留出更少空间
            plt.subplots_adjust(right=0.92, wspace=0.06, hspace=0.25) 
            
            # 保存
            fig_dir.mkdir(parents=True, exist_ok=True)
            postfix = '_postprocessed' if use_postprocess else ''
            output_path = fig_dir / f'exp2_{dataset}_{acc_type}_accuracy.pdf' # 建议存为PDF矢量图
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # 为了预览也保存一份PNG
            # output_path_png = fig_dir / f'exp2_{dataset}_{acc_type}_accuracy{postfix}.png'
            # plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            
            print(f"✅ Exp2 Plot Saved: {output_path}")
            plt.close()


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验结果可视化工具")
    
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="数据目录")
    parser.add_argument("--tail-epochs", type=int, default=DEFAULT_TAIL_EPOCHS, help="统计轮次")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help="图片保存目录")
    
    parser.add_argument("--plot1", action="store_true", help="绘制图1: Exp1噪声折线图")
    parser.add_argument("--plot2", action="store_true", help="绘制图2: Exp2柱状图（后处理数据）")
    parser.add_argument("--all", action="store_true", help="绘制所有图表")
    parser.add_argument("--no-postprocess", action="store_true", 
                       help="禁用后处理（仅对plot1有效，默认启用后处理）")
    
    args = parser.parse_args()
    
    if not (args.plot1 or args.plot2 or args.all):
        print("⚠️  未指定要绘制的图表，使用 --all 绘制所有图表，或使用 --plot1/--plot2 选择特定图表")
        args.all = True
    
    use_postprocess = not args.no_postprocess  # 默认启用后处理
    
    if args.all or args.plot1:
        print("\n📊 正在绘制图1: Exp1噪声折线图...")
        plot_exp1_noise_linecharts(args.output_dir, args.tail_epochs, args.fig_dir)
    
    if args.all or args.plot2:
        print("\n📊 正在绘制图2: Exp2柱状图（后处理数据）...")
        plot_exp2_bar_charts(args.output_dir, args.tail_epochs, args.fig_dir, use_postprocess)
    
    print(f"\n✅ 所有图表已保存到: {args.fig_dir}")


if __name__ == "__main__":
    main()


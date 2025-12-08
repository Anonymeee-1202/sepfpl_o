import argparse
import pickle
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.collections import PolyCollection


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
DEFAULT_OUTPUT_DIR = Path.home() / 'code/sepfpl/outputs'
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
    """
    查找文件，支持旧格式和新格式（包含 topk 和 rdp_p 参数）。
    
    对于 sepfpl 相关方法，文件名可能包含 topk 和 rdp_p 参数，格式为：
    acc_sepfpl_8_0.4_topk8_rdp1_01_1_10.pkl
    """
    import glob
    
    # 首先尝试精确匹配（向后兼容）
    possible_names = [f'{pattern_base}.pkl', f'{pattern_base}_10.pkl']
    for name in possible_names:
        file_path = base_dir / name
        if file_path.exists():
            return file_path
    
    # 如果精确匹配失败，使用 glob 模式匹配（支持包含 topk 和 rdp_p 的文件名）
    # 模式：pattern_base 后面可能跟 _topk*_rdp* 或 _rdp*_topk*，然后是 _num_users.pkl
    glob_patterns = [
        f'{pattern_base}.pkl',  # 旧格式
        f'{pattern_base}_*.pkl',  # 包含额外参数的新格式
    ]
    
    for pattern in glob_patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            # 返回第一个匹配的文件
            return matches[0]
    
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
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],  # 字体回退
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
        ('EXPERIMENT_1_STANDARD', 'exp1_simple'),
        ('EXPERIMENT_1_EXTENSION', 'exp1_hard'),
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

        if config_key == 'EXPERIMENT_1_STANDARD':
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

def plot_ablation_study(save_name="ablation_study_comparison"):
    """
    绘制顶刊学术风格的消融实验分组柱状图。
    特点：Times New Roman字体、大字号、专业配色、纹理填充、去边框。
    """
    
    # ================= 0. 全局样式设置 (Academic Style) =================
    # 使用字典更新 rcParams，确保无需安装额外包即可获得学术风格
    academic_params = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],  # 字体回退
        'font.size': 14 + 4,
        'axes.labelsize': 16 * 1.2,
        'axes.titlesize': 18 * 1.2,
        'xtick.labelsize': 14 * 1.2,
        'ytick.labelsize': 14 * 1.2,
        'legend.fontsize': 20,
        'figure.titlesize': 20 * 1.2,
        'axes.linewidth': 1.5,   # 坐标轴线变粗
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'lines.linewidth': 1.5,  # 误差棒变粗
        'mathtext.fontset': 'stix', # 数学公式字体与 Times 更搭
    }
    plt.rcParams.update(academic_params)

    # ================= 1. 数据准备 =================
    data = {
        "Caltech-101": {
            "Local Accuracy": {
                "Baseline":     [92.57, 91.73, 87.74],
                "w/ TA":        [94.34, 93.76, 88.88],
                "w/ SE":        [94.96, 94.32, 89.08],
                "SepFPL (Ours)":[95.42, 94.52, 90.46]
            },
            "Neighbor Accuracy": {
                "Baseline":     [91.93, 91.26, 87.37],
                "w/ TA":        [92.81, 91.46, 88.65],
                "w/ SE":        [92.86, 92.85, 89.30],
                "SepFPL (Ours)":[93.40, 92.86, 89.77]
            },
            "Local Std": {
                "Baseline":     [0.95, 1.29, 1.27],
                "w/ TA":        [0.34, 0.41, 1.11],
                "w/ SE":        [0.40, 0.42, 0.94],
                "SepFPL (Ours)":[0.72, 0.44, 0.94]
            },
            "Neighbor Std": {
                "Baseline":     [0.61, 0.70, 1.00],
                "w/ TA":        [0.60, 0.63, 0.89],
                "w/ SE":        [0.51, 0.29, 1.46],
                "SepFPL (Ours)":[0.37, 0.35, 1.05]
            }
        },
        "Stanford Dogs": {
            "Local Accuracy": {
                "Baseline":     [59.94, 58.29, 54.95],
                "w/ TA":        [60.08, 58.95, 55.00],
                "w/ SE":        [62.40, 61.17, 56.60],
                "SepFPL (Ours)":[64.53, 63.36, 56.71]
            },
            "Neighbor Accuracy": {
                "Baseline":     [59.35, 58.59, 53.83],
                "w/ TA":        [59.50, 58.84, 53.93],
                "w/ SE":        [60.77, 60.46, 55.16],
                "SepFPL (Ours)":[61.92, 61.16, 55.97]
            },
            "Local Std": {
                "Baseline":     [1.04, 0.90, 0.56],
                "w/ TA":        [0.78, 0.61, 0.92],
                "w/ SE":        [0.88, 0.40, 1.18],
                "SepFPL (Ours)":[0.95, 1.05, 1.26]
            },
            "Neighbor Std": {
                "Baseline":     [0.81, 0.59, 0.89],
                "w/ TA":        [0.75, 0.91, 1.11],
                "w/ SE":        [0.73, 0.72, 0.73],
                "SepFPL (Ours)":[0.50, 0.32, 0.81]
            }
        }
    }

    # ================= 2. 绘图配置 =================
    datasets = ["Caltech-101", "Stanford Dogs"]
    metrics = ["Local Accuracy", "Neighbor Accuracy"]
    epsilon_labels = ["0.4", "0.1", "0.01"]
    # 统一 Key 名称以匹配数据
    methods = ["Baseline", "w/ TA", "w/ SE", "SepFPL (Ours)"]
    
    # --- 学术配色方案 (Color Palette) ---
    # 1. 灰色系 (Baseline): 低调对比
    # 2. 蓝色系 (TA): 冷色调
    # 3. 绿色系 (SE): 冷色调
    # 4. 红色/橙色系 (Ours): 暖色调，高亮突出
    colors = ['#E0E0E0', '#99C1C2', '#8DA0CB', '#FC8D62'] 
    
    # --- 纹理填充 (Hatching) ---
    # 增加黑白打印时的辨识度
    # '/' = 斜线, '.' = 点, 'x' = 交叉, '' = 无
    hatches = ['///', '...', 'xx', ''] 

    x = np.arange(len(epsilon_labels))
    width = 0.2 

    # 初始化画布：2行2列，增加 DPI 保证清晰度
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, dpi=300)

    # ================= 3. 循环绘图 =================
    for row_idx, dataset in enumerate(datasets):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # 数据提取
            y_data = data[dataset][metric]
            std_key = "Local Std" if metric == "Local Accuracy" else "Neighbor Std"
            y_err = data[dataset][std_key]
            
            # 绘制柱子
            for i, method in enumerate(methods):
                offset = (i - 1.5) * width
                
                # 图例 Label 仅在第一个子图设置
                label = method if (row_idx == 0 and col_idx == 0) else ""
                
                # 绘制柱状图
                bars = ax.bar(x + offset, y_data[method], width, 
                              label=label,
                              color=colors[i], 
                              edgecolor='black', # 黑色边框
                              linewidth=1.2,     # 边框宽度
                              alpha=1.0,         # 不透明
                              yerr=y_err[method], 
                              capsize=4,         # 误差棒帽子宽度
                              error_kw={'elinewidth': 1.5, 'ecolor': '#333333'}, # 误差棒样式
                              zorder=3)          # 确保柱子在网格线之上
                
                # 应用纹理 (Hatching)
                # 注意：matplotlib 的 hatch 颜色默认随 edgecolor，
                # 这里我们保持黑色边框，纹理也是黑色的
                for bar in bars:
                    bar.set_hatch(hatches[i])

            # --- 样式微调 ---
            # 标题与坐标轴
            ax.set_title(f"{dataset} - {metric}", fontweight='bold', pad=12)
            
            if row_idx == 1:
                ax.set_xlabel(r"Privacy Budget ($\epsilon$)", fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(epsilon_labels)
            
            if col_idx == 0:
                ax.set_ylabel("Accuracy (%)", fontweight='bold')

            # --- 核心美化：网格与边框 ---
            # 仅保留 Y 轴网格，虚线，灰色，置于底层
            ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray', zorder=0)
            
            # 移除顶部和右侧边框 (Despine)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # 加粗左侧和底部边框
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            # --- Y轴范围动态调整 ---
            # 留出一点头部空间给误差棒
            if dataset == "Caltech-101":
                ax.set_ylim(85, 99) 
            else:
                ax.set_ylim(45, 70)

    # ================= 4. 全局图例与保存 =================
    # 获取图例句柄
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # 在顶部居中放置图例，无边框，背景透明
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
               ncol=4, frameon=False, columnspacing=1.5)

    plt.tight_layout()
    # 调整顶部边距，防止标题被图例遮挡
    # 减少上下子图间距(hspace)，增加图和图例之间的间距(top降低)
    plt.subplots_adjust(top=0.88, hspace=0.15, wspace=0.15) 

    # 路径处理
    save_dir = Path("figures") # 或者是 DEFAULT_FIG_DIR
    save_dir.mkdir(exist_ok=True)
    
    pdf_path = save_dir / f"{save_name}.pdf"
    
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"✅ 学术图表已生成:\n - {pdf_path}")
    
    plt.close()



    """
    绘制参数敏感性分析的折线图 (Line Chart)。
    包含三个子图：Rank, TopK, Schedule Factor (p)。
    采用顶刊学术风格。
    """

    # ================= 0. 全局样式设置 (Academic Style) =================
    academic_params = {
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'serif'],  # 字体回退
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12 + 4,
        'ytick.labelsize': 12 + 4,
        'legend.fontsize': 14,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'mathtext.fontset': 'stix',
    }
    plt.rcParams.update(academic_params)

    # ================= 1. 数据准备 =================
    # 将数据组织为字典，方便循环处理
    # 注意：这里使用的是 Oxford Flowers 的数据
    results = {
        "Rank": {
            "x": [1, 2, 4, 8, 16],
            "y_eps_01":  [69.86, 70.76, 67.66, 70.14, 69.85], # epsilon=0.1
            "y_eps_001": [66.79, 66.42, 66.08, 66.08, 65.63], # epsilon=0.01
            "xlabel": "Rank ($r$)",
            "title": "(a) Impact of Rank",
            "xticks": [1, 2, 4, 8, 16] # 强制显示这些刻度
        },
        "TopK": {
            "x": [2, 4, 6, 8],
            "y_eps_01":  [70.61, 70.41, 70.67, 70.14],
            "y_eps_001": [66.47, 66.28, 65.58, 66.08],
            "xlabel": "TopK ($K$)",
            "title": "(b) Impact of TopK",
            "xticks": [2, 4, 6, 8]
        },
        "P_Factor": {
            "x": [0, 0.2, 0.5, 1.0],
            "y_eps_01":  [70.50, 70.14, 69.89, 69.61],
            "y_eps_001": [67.08, 66.08, 65.07, 60.65],
            "xlabel": r"Schedule Factor ($p_\chi$)",
            "title": r"(c) Impact of $p_\chi$",
            "xticks": [0, 0.2, 0.5, 1.0]
        }
    }

    # ================= 2. 绘图配置 =================
    # 配色：蓝色(0.1) 和 红色(0.01)
    colors = {'eps_01': '#377eb8', 'eps_001': '#e41a1c'}
    markers = {'eps_01': 'o', 'eps_001': 's'} # 圆圈和方块
    linestyles = {'eps_01': '-', 'eps_001': '--'} # 实线和虚线

    # 初始化画布：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    keys = ["Rank", "TopK", "P_Factor"]

    # ================= 3. 循环绘图 =================
    for i, key in enumerate(keys):
        ax = axes[i]
        data = results[key]
        x_vals = data["x"]

        # --- 绘制线条 ---
        # Line 1: epsilon = 0.1
        ax.plot(x_vals, data["y_eps_01"], 
                color=colors['eps_01'], 
                marker=markers['eps_01'], 
                linestyle=linestyles['eps_01'],
                label=r'$\epsilon=0.1$' if i == 1 else "") # 标签仅加一次用于生成图例

        # Line 2: epsilon = 0.01
        ax.plot(x_vals, data["y_eps_001"], 
                color=colors['eps_001'], 
                marker=markers['eps_001'], 
                linestyle=linestyles['eps_001'],
                label=r'$\epsilon=0.01$' if i == 1 else "")

        # --- 样式调整 ---
        ax.set_title(data["title"], fontweight='bold', pad=12)
        ax.set_xlabel(data["xlabel"], fontweight='bold')
        
        # 仅在第一个图显示 Y 轴标签
        if i == 0:
            ax.set_ylabel("Local Accuracy (%)", fontweight='bold')

        # 设置刻度
        ax.set_xticks(data["xticks"])
        
        # 网格与边框 (Academic Style)
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        
        # 去掉上方和右侧边框 (Despine)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 适当调整 Y 轴范围以留出空间
        if key == "P_Factor":
             ax.set_ylim(58, 72) # P因子下降较明显，调整范围
        else:
             ax.set_ylim(64, 72)

    # ================= 4. 全局图例与保存 =================
    # 提取图例句柄 (从中间的子图提取)
    handles, labels = axes[1].get_legend_handles_labels()
    
    # 在顶部居中放置图例
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=2, frameon=False, fontsize=14)

    plt.tight_layout()
    # 调整布局防止标题被图例遮挡
    plt.subplots_adjust(top=0.85, wspace=0.25)

    # 保存路径处理
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)
    
    pdf_path = save_dir / f"{save_name}.pdf"
    png_path = save_dir / f"{save_name}.png"
    
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    
    print(f"✅ 敏感性分析图表已生成:\n - {pdf_path}")

    if show_plot:
        plt.show()
    else:
            plt.close()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import matplotlib.ticker as ticker

# ================= 0. 配置与样式 =================
def set_academic_style():
    """配置学术风格的绘图参数"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'mathtext.fontset': 'stix',
        'figure.titlesize': 18,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

# ================= 1. 数据封装 =================
def get_data():
    """返回封装好的实验数据"""
    
    # 隐私预算标签 (用于Y轴)
    # Group A: Rank & TopK (含 Noise=0)
    eps_labels_A = [r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon=0.4$', r'$\epsilon=\infty$'] 
    # Group B: p (不含 Noise=0)
    eps_labels_B = [r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon=0.4$']

    # 颜色配置 (用于不同 Epsilon)
    # 使用渐变色：深蓝 -> 蓝 -> 浅蓝 -> 紫(无噪)
    colors_A = ['#08519c', '#3182bd', '#6baed6', '#9e9ac8'] 
    colors_B = ['#08519c', '#3182bd', '#6baed6']

    # --- Rank Data ---
    rank_x = [1, 2, 4, 8, 16]
    # [Dogs_Loc, Dogs_Ngh, Flowers_Loc, Flowers_Ngh]
    # Data order: [Noise=0, 0.4, 0.1, 0.01] -> Reorder to [0.01, 0.1, 0.4, 0] for plotting (Back to Front)
    # Raw data order from user: 0, 0.4, 0.1, 0.01
    # Target plotting order (y=0..3): 0.01, 0.1, 0.4, 0
    _rank_raw = [
        [[61.15, 61.46, 61.21, 59.01, 60.43], [58.35, 59.60, 60.32, 60.61, 61.19], [58.28, 60.28, 60.10, 59.14, 61.10], [56.24, 58.13, 57.32, 56.92, 56.84]], # Dogs Loc
        [[61.97, 61.60, 60.55, 59.40, 59.94], [58.62, 59.41, 59.67, 60.19, 60.76], [57.57, 59.55, 59.94, 59.62, 59.77], [56.72, 57.27, 56.29, 56.15, 56.71]], # Dogs Ngh
        [[69.81, 71.43, 69.16, 70.03, 68.85], [69.46, 70.95, 68.95, 70.99, 70.60], [69.86, 70.76, 67.66, 70.14, 69.85], [66.79, 66.42, 66.08, 66.08, 65.63]], # Flowers Loc
        [[70.65, 71.24, 69.54, 69.54, 69.47], [69.25, 70.95, 68.63, 71.13, 70.36], [69.81, 70.37, 66.64, 69.71, 69.92], [66.27, 65.94, 65.66, 65.77, 66.85]]  # Flowers Ngh
    ]
    # Reorder function: [0, 1, 2, 3] -> [3, 2, 1, 0] to match labels [0.01, 0.1, 0.4, 0]
    # User Raw: 0(idx0), 0.4(idx1), 0.1(idx2), 0.01(idx3)
    # Target:   0.01, 0.1, 0.4, 0
    reorder_idx_A = [3, 2, 1, 0] 
    rank_data = [[dataset[i] for i in reorder_idx_A] for dataset in _rank_raw]

    # --- TopK Data ---
    topk_x = [2, 4, 6, 8]
    _topk_raw = [
        [[59.16, 59.22, 58.87, 59.01], [60.67, 60.49, 60.10, 60.61], [59.65, 59.85, 60.04, 59.14], [58.39, 56.50, 57.49, 56.92]],
        [[59.70, 60.34, 59.74, 59.40], [60.40, 60.16, 60.13, 60.19], [59.55, 59.81, 59.51, 59.62], [58.09, 56.19, 57.01, 56.15]],
        [[70.27, 70.17, 69.88, 70.03], [71.27, 71.27, 70.78, 70.99], [70.61, 70.41, 70.67, 70.14], [66.47, 66.28, 65.58, 66.08]],
        [[69.67, 69.53, 69.30, 69.54], [70.95, 70.66, 70.19, 71.13], [69.96, 70.13, 70.09, 69.71], [66.36, 66.01, 65.58, 65.77]]
    ]
    topk_data = [[dataset[i] for i in reorder_idx_A] for dataset in _topk_raw]

    # --- P Data ---
    p_x = [0, 0.2, 0.5, 1]
    # User Raw: 0.4(idx0), 0.1(idx1), 0.01(idx2)
    # Target:   0.01, 0.1, 0.4
    reorder_idx_B = [2, 1, 0]
    _p_raw = [
        [[60.66, 60.61, 59.86, 60.27], [59.56, 59.14, 60.06, 60.36], [58.28, 56.92, 56.47, 57.22]],
        [[60.18, 60.19, 59.77, 59.68], [59.73, 59.62, 59.91, 59.91], [58.52, 56.15, 56.03, 55.96]],
        [[70.65, 70.99, 70.91, 71.19], [70.50, 70.14, 69.89, 69.61], [67.08, 66.08, 65.07, 60.65]],
        [[70.36, 71.13, 70.35, 70.21], [69.81, 69.71, 70.10, 69.24], [67.00, 65.77, 64.97, 59.53]]
    ]
    p_data = [[dataset[i] for i in reorder_idx_B] for dataset in _p_raw]

    return {
        "params": [
            {"data": rank_data, "x": rank_x, "xlabel": "Rank ($r$)", "title": "(a) Impact of Rank", "eps_labels": eps_labels_A, "colors": colors_A},
            {"data": topk_data, "x": topk_x, "xlabel": "TopK ($K$)", "title": "(b) Impact of TopK", "eps_labels": eps_labels_A, "colors": colors_A},
            {"data": p_data,   "x": p_x,    "xlabel": r"Schedule Factor ($p$)", "title": "(c) Impact of $p$", "eps_labels": eps_labels_B, "colors": colors_B}
        ],
        "datasets": [
            {"name": "Stanford Dogs", "indices": [0, 1], "zlim": (55, 63)},
            {"name": "Oxford Flowers", "indices": [2, 3], "zlim": (58, 72)}
        ]
    }

# ================= 2. 绘图核心函数 =================
def plot_ribbon_subplot(ax, x_vals, dataset_data, eps_labels, colors, xlabel, title, zlim, show_zlabel=True):
    """
    在给定的 3D 轴上绘制单个参数的 Ribbon 图。
    dataset_data: shape [num_eps, len(x)]，包含 Local 和 Neighbor 两种数据? 
    不，这里传入的是单个数据集的单个指标数据列表。
    为了在同一张图显示 Local 和 Neighbor，我们需要处理两条带子。
    show_zlabel: 是否显示z轴标签，默认True
    """
    
    # 调整视角
    ax.view_init(elev=20, azim=-70)
    
    num_eps = len(eps_labels)
    xs = np.arange(len(x_vals))
    
    # 辅助函数：将颜色转换为 RGB，用于调整亮度和色调
    def adjust_color_for_neighbor(color, lighten_factor=0.25, shift_hue=0.05):
        """调整颜色用于 Neighbor 线条：变浅并略微向红色调偏移"""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        # 向白色方向混合（变浅）
        lightened = tuple(1 - (1 - c) * (1 - lighten_factor) for c in rgb)
        # 略微增加红色分量，使颜色更暖
        adjusted = (min(1.0, lightened[0] + shift_hue), lightened[1], lightened[2])
        return adjusted
    
    # 辅助函数：绘制单条 Ribbon
    def add_ribbon(y_index, z_values, color, label=None, linestyle='-', is_neighbor=False):
        # 1. 顶部线条
        line_color = adjust_color_for_neighbor(color) if is_neighbor else color
        line_width = 2.2 if not is_neighbor else 2.0  # Local 线条更粗，更突出
        ax.plot(xs, [y_index]*len(xs), z_values, 
                color=line_color, linewidth=line_width, linestyle=linestyle,
                marker='o', markersize=5 if not is_neighbor else 4, 
                markerfacecolor='white', markeredgecolor=line_color, markeredgewidth=1.5,
                zorder=10 + y_index, label=label)
        
        # 2. 填充面 (PolyCollection) - 仅用于 Local
        if not is_neighbor:
            verts = []
            # 底部基准线 (z=zmin)
            z_min = zlim[0]
            polygon = [(x, z_min) for x in xs] + [(x, z) for x, z in zip(xs, z_values)][::-1]
            verts.append(polygon)
            
            # 使用稍深的颜色用于填充，增加对比度
            poly = PolyCollection(verts, facecolors=color, edgecolors=color, 
                                 alpha=0.4, linewidths=0.5) # 降低透明度，添加边框
            ax.add_collection3d(poly, zs=y_index, zdir='y')
    
    # 获取 Local 和 Neighbor 数据
    # dataset_data 是个 list，包含 [loc_data_list, ngh_data_list]
    loc_data_list = dataset_data[0]
    ngh_data_list = dataset_data[1]

    for i in range(num_eps):
        # 颜色：越靠前（epsilon 越小）颜色越深，或者反之
        c = colors[i]
        
        # 绘制 Local Ribbon（实线，带填充）
        add_ribbon(i, loc_data_list[i], c, label=f"{eps_labels[i]}" if i==0 else None, 
                  linestyle='-', is_neighbor=False)
        
        # 绘制 Neighbor Line（虚线，无填充，使用稍浅的颜色）
        add_ribbon(i, ngh_data_list[i], c, linestyle='--', is_neighbor=True)

    # --- 坐标轴设置 ---
    # X轴
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.set_xlabel(xlabel, labelpad=5, fontweight='bold')
    
    # Y轴
    ax.set_yticks(np.arange(num_eps))
    ax.set_yticklabels(eps_labels, verticalalignment='baseline', horizontalalignment='left')
    # 调整 Y 轴标签角度
    plt.setp(ax.get_yticklabels(), fontsize=12)
    
    # Z轴
    ax.set_zlim(zlim)
    if show_zlabel:
        ax.set_zlabel("Accuracy (%)", fontweight='bold', labelpad=5)
    
    # 标题
    # ax.set_title(title, y=1.05, fontweight='bold')
    
    # 优化面板显示 - 使用更清晰的背景色
    # 使用淡蓝色背景，提高对比度
    pane_color = '#f0f0f5'  # 淡蓝灰色
    
    ax.xaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor(pane_color)
    ax.xaxis.pane.set_alpha(0.3)
    
    ax.yaxis.pane.fill = True
    ax.yaxis.pane.set_facecolor(pane_color)
    ax.yaxis.pane.set_alpha(0.3)
    
    ax.zaxis.pane.fill = True
    ax.zaxis.pane.set_facecolor(pane_color)
    ax.zaxis.pane.set_alpha(0.3)
    
    # 设置坐标轴颜色，增强可见性
    ax.xaxis.line.set_color('#666666')
    ax.yaxis.line.set_color('#666666')
    ax.zaxis.line.set_color('#666666')
    
    ax.grid(False) # 移除默认网格
    
    # 手动添加 Z 轴网格线 (仅在背板) - 使用更明显的颜色
    for z in np.linspace(zlim[0], zlim[1], 5):
        ax.plot([xs[0], xs[-1]], [num_eps-1, num_eps-1], [z, z], 
                color='#999999', alpha=0.3, linestyle='--', linewidth=0.8)


def plot_sensitivity_analysis(save_name="sensitivity_analysis_refined", show_plot=True):
    set_academic_style()
    data_pack = get_data()
    
    # 为每个数据集生成一张大图 (1行3列)
    for ds_conf in data_pack["datasets"]:
        ds_name = ds_conf["name"]
        indices = ds_conf["indices"] # [loc_idx, ngh_idx]
        zlim = ds_conf["zlim"]
        
        fig = plt.figure(figsize=(18, 6))
        # fig.suptitle(f"Parameter Sensitivity on {ds_name}", fontsize=20, y=0.95)
        
        num_params = len(data_pack["params"])
        for i, param_conf in enumerate(data_pack["params"]):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            
            # 提取该参数下，该数据集的 Local 和 Neighbor 数据
            # param_conf["data"] 结构是 [Dogs_Loc, Dogs_Ngh, Flowers_Loc, Flowers_Ngh]
            # 我们需要取出 indices 对应的两个列表
            current_ds_data = [param_conf["data"][indices[0]], param_conf["data"][indices[1]]]
            
            # 只有最右边的子图（最后一个）显示z轴标签
            show_zlabel = (i == num_params - 1)
            
            plot_ribbon_subplot(
                ax, 
                param_conf["x"], 
                current_ds_data, 
                param_conf["eps_labels"], 
                param_conf["colors"], 
                param_conf["xlabel"], 
                param_conf["title"],
                zlim,
                show_zlabel=show_zlabel
            )
            
            # 添加自定义图例 (仅在第一个子图)
            if i == 0:
                # 创建虚拟句柄用于图例
                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch
                legend_elements = [
                    Line2D([0], [0], color='black', lw=2, label='Local Acc.'),
                    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Neighbor Acc.'),
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.1, 1.0), fontsize=10, frameon=False)

        plt.subplots_adjust(left=0.1, right=0.90, wspace=0.01)
        
        save_path = Path("figures") / f"{save_name}_{ds_name.lower().replace(' ', '_')}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()



# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验结果可视化工具")
    
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="数据目录")
    parser.add_argument("--tail-epochs", type=int, default=DEFAULT_TAIL_EPOCHS, help="统计轮次")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help="图片保存目录")
    
    parser.add_argument("--plot1", action="store_true", help="绘制图1: Exp1噪声折线图")
    parser.add_argument("--ablation", action="store_true", help="绘制消融实验分组柱状图")
    parser.add_argument("--sensitivity", action="store_true", help="绘制参数敏感性分析折线图")
    
    args = parser.parse_args()
    
    if not (args.plot1 or args.ablation or args.sensitivity):
        print("⚠️  未指定要绘制的图表，使用 --plot1 绘制Exp1噪声折线图，或使用 --ablation 绘制消融实验图，或使用 --sensitivity 绘制敏感性分析图")
        args.plot1 = True
    
    if args.plot1:
        print("\n📊 正在绘制图1: Exp1噪声折线图...")
        plot_exp1_noise_linecharts(args.output_dir, args.tail_epochs, args.fig_dir)
    
    if args.ablation:
        print("\n📊 正在绘制消融实验分组柱状图...")
        plot_ablation_study()
    
    if args.sensitivity:
        print("\n📊 正在绘制参数敏感性分析折线图...")
        plot_sensitivity_analysis()
    
    if args.plot1:
        print(f"\n✅ 所有图表已保存到: {args.fig_dir}")


if __name__ == "__main__":
    main()


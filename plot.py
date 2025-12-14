import argparse
import pickle
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# 导入共享的数据工具函数
from utils.data_utils import (
    DEFAULT_TAIL_EPOCHS,
    tail_values,
    load_metrics,
    find_output_file,
    postprocess_results,
    extract_value,
    read_data,
    read_scheme,
)

# ========== 全局配置 ==========
# 注意：默认使用outputs目录（与table.py一致），如果数据在p_outputs，请使用--output-dir参数指定
DEFAULT_OUTPUT_DIR = Path.home() / 'code/sepfpl/outputs'
DEFAULT_FIG_DIR = Path('figures')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


# ========== 数据读取函数（复用 data_utils 的逻辑） ==========

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


# ================= 敏感性分析3D Ribbon图绘制器 =================
class SensitivityAnalysisPlotter:
    """
    参数敏感性分析3D Ribbon图绘制器
    
    封装了绘制参数敏感性分析所需的所有功能，包括：
    - 学术风格配置
    - 数据准备
    - 3D Ribbon子图绘制
    - 完整图表生成
    """
    
    @staticmethod
    def _set_academic_style():
        """配置学术风格的绘图参数"""
        plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
        'font.size': 12 + 4,
        'axes.labelsize': 14 + 2,
        'axes.titlesize': 16,
        'xtick.labelsize': 12 + 4,
        'ytick.labelsize': 12 + 4,
        'legend.fontsize': 16,
        'mathtext.fontset': 'stix',
        'figure.titlesize': 18,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
        })
    
    @staticmethod
    def _get_data():
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
            [[60.61, 60.66, 59.86, 60.27], [59.56, 60.36, 60.06, 59.14], [56.92, 58.28, 56.47, 57.22]],
            [[60.18, 60.19, 59.77, 59.68], [59.73, 59.91, 59.62, 59.91], [56.15, 58.52, 56.03, 55.96]],
            [[70.65, 71.19, 70.99, 70.91], [70.14, 70.50, 69.89, 69.61], [66.08, 67.08, 65.07, 60.65]],
            [[70.36, 71.13, 70.35, 70.21], [69.81, 70.10, 69.71, 69.24], [65.77, 67.00, 64.97, 59.53]]
        ]
        p_data = [[dataset[i] for i in reorder_idx_B] for dataset in _p_raw]

        return {
            "params": [
                {"data": rank_data, "x": rank_x, "xlabel": "Rank ($r$)", "title": "(a) Impact of Rank", "eps_labels": eps_labels_A, "colors": colors_A},
                {"data": topk_data, "x": topk_x, "xlabel": "TopM ($M$)", "title": "(b) Impact of TopK", "eps_labels": eps_labels_A, "colors": colors_A},
                {"data": p_data,   "x": p_x,    "xlabel": r"Schedule Factor ($p$)", "title": "(c) Impact of $p$", "eps_labels": eps_labels_B, "colors": colors_B}
            ],
            "datasets": [
                {"name": "Stanford Dogs", "indices": [0, 1], "zlim": (55, 63)},
                {"name": "Oxford Flowers", "indices": [2, 3], "zlim": (58, 72)}
            ]
        }
    
    @staticmethod
    def _adjust_color_for_neighbor(color, lighten_factor=0.25, shift_hue=0.05):
        """调整颜色用于 Neighbor 线条：变浅并略微向红色调偏移"""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        # 向白色方向混合（变浅）
        lightened = tuple(1 - (1 - c) * (1 - lighten_factor) for c in rgb)
        # 略微增加红色分量，使颜色更暖
        adjusted = (min(1.0, lightened[0] + shift_hue), lightened[1], lightened[2])
        return adjusted
    
    @staticmethod
    def _plot_ribbon_subplot(ax, x_vals, dataset_data, eps_labels, colors, xlabel, title, zlim, show_zlabel=True):
        """
        在给定的 3D 轴上绘制单个参数的 Ribbon 图。
        
        Args:
            ax: 3D 坐标轴对象
            x_vals: X轴数值列表
            dataset_data: [loc_data_list, ngh_data_list]，包含 Local 和 Neighbor 数据
            eps_labels: 隐私预算标签列表
            colors: 颜色列表
            xlabel: X轴标签
            title: 子图标题
            zlim: Z轴范围 (min, max)
            show_zlabel: 是否显示z轴标签，默认True
        """
        # 调整视角
        ax.view_init(elev=20, azim=-70)
        
        num_eps = len(eps_labels)
        xs = np.arange(len(x_vals))
        
        # 辅助函数：绘制单条 Ribbon
        def add_ribbon(y_index, z_values, color, label=None, linestyle='-', is_neighbor=False):
            # 1. 顶部线条
            line_color = SensitivityAnalysisPlotter._adjust_color_for_neighbor(color) if is_neighbor else color
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
        ax.set_xlabel(xlabel, labelpad=8, fontweight='bold')
        
        # Y轴
        ax.set_yticks(np.arange(num_eps))
        ax.set_yticklabels(eps_labels, verticalalignment='baseline', horizontalalignment='left')
        # 调整 Y 轴标签角度
        plt.setp(ax.get_yticklabels(), fontsize=12 + 4)
        
        # Z轴
        ax.set_zlim(zlim)
        if show_zlabel:
            ax.set_zlabel("Accuracy (%)", fontweight='bold', labelpad=8)
        
        # 优化面板显示 - 使用更清晰的背景色
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

    @classmethod
    def plot(cls, save_name="sensitivity_analysis_refined", show_plot=True):
        """
        生成参数敏感性分析的3D Ribbon图
        
        Args:
            save_name: 保存文件名前缀
            show_plot: 是否显示图表（默认True，False则只保存）
        """
        cls._set_academic_style()
        data_pack = cls._get_data()
        
        # 为每个数据集生成一张大图 (1行3列)
        for ds_conf in data_pack["datasets"]:
            ds_name = ds_conf["name"]
            indices = ds_conf["indices"] # [loc_idx, ngh_idx]
            zlim = ds_conf["zlim"]
            
            fig = plt.figure(figsize=(18, 6))
            
            num_params = len(data_pack["params"])
            for i, param_conf in enumerate(data_pack["params"]):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                
                # 提取该参数下，该数据集的 Local 和 Neighbor 数据
                current_ds_data = [param_conf["data"][indices[0]], param_conf["data"][indices[1]]]
                
                # 只有最右边的子图（最后一个）显示z轴标签
                show_zlabel = (i == num_params - 1)
                
                cls._plot_ribbon_subplot(
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
                
                # 添加自定义图例 (仅在最后一个子图)
                if i == num_params - 1:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='black', lw=2, label='Local Acc.'),
                        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Neighbor Acc.'),
                    ]
                    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=18, frameon=False)

            plt.subplots_adjust(left=0.1, right=0.90, wspace=0.01)
            
            save_path = Path("figures") / f"{save_name}_{ds_name.lower().replace(' ', '_')}.pdf"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()

# ================= MIA分析绘图类 =================
class MiaAnalysisPlotter:
    """
    MIA分析绘图类，用于绘制包含3个子图的综合分析图：
    1. Local Accuracy vs Privacy Budget
    2. Neighbor Accuracy vs Privacy Budget
    3. MIA Attack Success Rate vs Privacy Budget
    """
    
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR, 
                 tail_epochs: int = DEFAULT_TAIL_EPOCHS,
                 fig_dir: Path = DEFAULT_FIG_DIR):
        """
        初始化绘图器
        
        Args:
            output_dir: 数据目录
            tail_epochs: 统计最后N轮的平均值
            fig_dir: 图片保存目录
        """
        self.output_dir = output_dir
        self.tail_epochs = tail_epochs
        self.fig_dir = fig_dir
        
        # 数据集标签映射（美化显示）
        self.dataset_labels = {
            'caltech-101': 'Caltech-101',
            'stanford_dogs': 'Stanford Dogs',
            'oxford_flowers': 'Oxford Flowers',
            'food-101': 'Food-101'
        }
        
        # 数据集颜色配置
        self.dataset_colors = {
            'caltech-101': '#1f77b4',      # Blue
            'oxford_flowers': '#2ca02c',   # Green
            'food-101': '#d62728',         # Red
            'stanford_dogs': '#ff7f0e'      # Orange
        }
        
        # 数据集标记配置
        self.dataset_markers = {
            'caltech-101': 'o',    # Circle
            'oxford_flowers': '^', # Triangle Up
            'food-101': 'D',       # Diamond
            'stanford_dogs': 's'   # Square
        }
        
        # 设置学术绘图风格
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
            'mathtext.fontset': 'stix',
            'font.size': 32,
            'axes.labelsize': 24,
            # 'axes.titlesize': 16,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'legend.fontsize': 24,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 4,
            'lines.markersize': 12,
        })
    
    def _load_exp1_data(self):
        """加载实验1 Standard的数据"""
        import pickle
        
        # 从配置获取实验参数
        if 'EXPERIMENT_1_STANDARD' not in EXPERIMENT_CONFIGS:
            print("❌ 错误: 找不到 EXPERIMENT_1_STANDARD 配置")
            return {}, {}, [], []
        
        config = EXPERIMENT_CONFIGS['EXPERIMENT_1_STANDARD']
        exp_name = config.get('exp_name', 'exp1-standard')
        datasets = config.get('dataset_list', [])
        noise_list = config.get('noise_list', [0.0, 0.4, 0.2, 0.1, 0.05, 0.01])
        rank = config.get('rank_list', [8])[0]
        seed_list = config.get('seed_list', [1])
        num_users = config.get('num_users_list', [10])[0]
        sepfpl_topk = config.get('sepfpl_topk', 8)
        rdp_p = config.get('rdp_p', 0.2)
        factorization = 'sepfpl'
        
        # 读取数据
        base_dir = self.output_dir / exp_name
        dataset_local_map = {}
        dataset_neighbor_map = {}
        
        for dataset in datasets:
            dataset_dir = base_dir / dataset
            if not dataset_dir.exists():
                print(f"⚠️  警告: 数据集目录不存在: {dataset_dir}")
                continue
            
            local_accs = []
            neighbor_accs = []
            
            for noise in noise_list:
                # 构建文件名模式
                if noise == int(noise):
                    noise_str = f'{float(noise):.1f}'
                else:
                    noise_str = f'{float(noise):g}'
                
                rdp_p_str = str(rdp_p)
                
                # 读取所有seed的数据并计算平均值
                per_seed_local = []
                per_seed_neighbor = []
                
                for seed in seed_list:
                    pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_{rdp_p_str}_{num_users}'
                    file_path = find_output_file(dataset_dir, pattern)
                    
                    if file_path and file_path.exists():
                        try:
                            with open(file_path, 'rb') as f:
                                data = pickle.load(f)
                            
                            # 数据格式可能是 [local_acc_list, neighbor_acc_list] 或 dict
                            if isinstance(data, list) and len(data) >= 2:
                                local_hist = data[0] if isinstance(data[0], list) else []
                                neighbor_hist = data[1] if isinstance(data[1], list) else []
                            elif isinstance(data, dict):
                                local_hist = data.get('local_acc', [])
                                neighbor_hist = data.get('neighbor_acc', [])
                            else:
                                local_hist, neighbor_hist = [], []
                            
                            # 获取最后 tail_epochs 轮的平均值
                            if local_hist:
                                tail_local = tail_values(local_hist, self.tail_epochs)
                                if tail_local:
                                    per_seed_local.extend(tail_local)
                            
                            if neighbor_hist:
                                tail_neighbor = tail_values(neighbor_hist, self.tail_epochs)
                                if tail_neighbor:
                                    per_seed_neighbor.extend(tail_neighbor)
                        except Exception as e:
                            print(f"⚠️  警告: 无法读取 {file_path}: {e}")
                
                # 计算该noise值下的平均准确率
                if per_seed_local:
                    local_accs.append(mean(per_seed_local))
                else:
                    local_accs.append(0.0)
                
                if per_seed_neighbor:
                    neighbor_accs.append(mean(per_seed_neighbor))
                else:
                    neighbor_accs.append(0.0)
            
            if local_accs or neighbor_accs:
                dataset_local_map[dataset] = local_accs
                dataset_neighbor_map[dataset] = neighbor_accs
        
        return dataset_local_map, dataset_neighbor_map, noise_list, datasets
    
    def _load_exp4_data(self):
        """加载实验4 MIA的数据"""
        import pickle
        
        # 从配置获取实验参数
        if 'EXPERIMENT_4_MIA' not in EXPERIMENT_CONFIGS:
            print("❌ 错误: 找不到 EXPERIMENT_4_MIA 配置")
            return {}, [], []
        
        config = EXPERIMENT_CONFIGS['EXPERIMENT_4_MIA']
        exp_name = config.get('exp_name', 'exp4-mia')
        datasets = config.get('dataset_list', [])
        noise_list = config.get('noise_list', [0.0, 0.4, 0.2, 0.1, 0.05, 0.01])
        
        # 读取数据
        base_dir = self.output_dir / exp_name
        dataset_acc_map = {}
        
        for dataset in datasets:
            dataset_dir = base_dir / dataset
            if not dataset_dir.exists():
                print(f"⚠️  警告: 数据集目录不存在: {dataset_dir}")
                continue
            
            accuracies = []
            for noise in noise_list:
                # 构建文件路径
                mia_acc_file = dataset_dir / f'mia_acc_{noise}.pkl'
                if mia_acc_file.exists():
                    try:
                        with open(mia_acc_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        if isinstance(data, dict):
                            avg_acc = data.get('average', 0.0)
                            accuracies.append(avg_acc * 100)  # 转换为百分比
                        elif isinstance(data, (int, float)):
                            accuracies.append(float(data) * 100)
                        else:
                            print(f"⚠️  警告: {mia_acc_file} 数据格式不正确")
                            accuracies.append(0.0)
                    except Exception as e:
                        print(f"⚠️  警告: 无法读取 {mia_acc_file}: {e}")
                        accuracies.append(0.0)
                else:
                    print(f"⚠️  警告: 文件不存在: {mia_acc_file}")
                    accuracies.append(0.0)
            
            if accuracies:
                dataset_acc_map[dataset] = accuracies
        
        return dataset_acc_map, noise_list, datasets
    
    def _plot_subplot(self, ax, acc_map, datasets, noise_list, ylabel, title=None, show_legend=False, y_lim=None):
        """绘制单个子图"""
        x_positions = np.arange(len(noise_list))
        x_tick_labels = ['$\infty$'] + [f'{n}' for n in noise_list[1:]]
        
        # 绘制每条折线
        for dataset in datasets:
            if dataset not in acc_map:
                continue
            
            accuracies = acc_map[dataset]
            label = self.dataset_labels.get(dataset, dataset)
            color = self.dataset_colors.get(dataset, '#333333')
            marker = self.dataset_markers.get(dataset, 'o')
            
            ax.plot(x_positions, accuracies,
                    marker=marker,
                    label=label,
                    color=color,
                    markeredgecolor='white',
                    markeredgewidth=1.5,
                    zorder=10)
        
        # 设置标题（如果提供）
        if title:
            ax.set_title(title, fontweight='bold', pad=12)
        ax.set_xlabel(r'Privacy Budget ($\epsilon$)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        
        # 设置X轴刻度
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels)
        
        # 网格和边框美化
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # 设置Y轴范围
        if y_lim is not None:
            ax.set_ylim(y_lim)
        else:
            all_accs = [acc for accs in acc_map.values() for acc in accs if acc > 0]
            if all_accs:
                min_acc = min(all_accs)
                max_acc = max(all_accs)
                ax.set_ylim(bottom=max(0, min_acc - 5), top=min(105, max_acc + 5))
        
        # 图例（只在需要时显示，且放在右侧）
        if show_legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    def plot(self):
        """绘制包含3个子图的综合分析图"""
        # 加载数据
        dataset_local_map, dataset_neighbor_map, noise_list_exp1, datasets_exp1 = self._load_exp1_data()
        dataset_mia_map, noise_list_exp4, datasets_exp4 = self._load_exp4_data()
        
        if not dataset_local_map and not dataset_neighbor_map:
            print("❌ 错误: 没有找到实验1的任何数据")
            return
        
        if not dataset_mia_map:
            print("❌ 错误: 没有找到实验4的任何数据")
            return
        
        # 确保两个实验使用相同的数据集和噪声列表
        datasets = list(set(datasets_exp1) & set(datasets_exp4))
        if not datasets:
            print("❌ 错误: 两个实验没有共同的数据集")
            return
        
        # 使用实验1的噪声列表（通常两个实验应该一致）
        noise_list = noise_list_exp1
        
        # 计算 Local 和 Neighbor 的共同Y轴范围
        all_local_accs = [acc for accs in dataset_local_map.values() for acc in accs if acc > 0]
        all_neighbor_accs = [acc for accs in dataset_neighbor_map.values() for acc in accs if acc > 0]
        all_acc_accs = [acc for accs in dataset_mia_map.values() for acc in accs if acc > 0]
        
        # 计算 Local 和 Neighbor 的共同范围
        combined_accs = all_local_accs + all_neighbor_accs
        if combined_accs:
            min_acc = min(combined_accs)
            max_acc = max(combined_accs)
            shared_y_lim = (max(0, min_acc - 5), min(105, max_acc + 5))
        else:
            shared_y_lim = None
        
        # 计算 MIA 的Y轴范围
        if all_acc_accs:
            min_mia = min(all_acc_accs)
            max_mia = max(all_acc_accs)
            mia_y_lim = (max(0, min_mia - 5), min(105, max_mia + 5))
        else:
            mia_y_lim = None
        
        # 创建包含3个子图的figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # 绘制第一个子图：Local Accuracy（无标题，使用共享Y轴范围）
        self._plot_subplot(
            axes[0], 
            dataset_local_map, 
            datasets, 
            noise_list,
            'Local Accuracy (%)',
            title=None,
            show_legend=False,
            y_lim=shared_y_lim
        )
        
        # 绘制第二个子图：Neighbor Accuracy（无标题，使用共享Y轴范围）
        self._plot_subplot(
            axes[1], 
            dataset_neighbor_map, 
            datasets, 
            noise_list,
            'Neighbor Accuracy (%)',
            title=None,
            show_legend=False,
            y_lim=shared_y_lim
        )
        
        # 绘制第三个子图：MIA Success Rate（无标题，包含图例）
        self._plot_subplot(
            axes[2], 
            dataset_mia_map, 
            datasets, 
            noise_list,
            'MIA Success Rate (%)',
            title=None,
            show_legend=True,
            y_lim=mia_y_lim
        )
        
        # 调整布局，为右侧图例留出空间
        plt.tight_layout(rect=[0, 0, 0.97, 1])
        
        # 保存图片
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.fig_dir / 'mia_analysis_combined.pdf'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"✅ MIA综合分析图已保存: {output_path}")
        plt.close()


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验结果可视化工具")
    
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="数据目录")
    parser.add_argument("--tail-epochs", type=int, default=DEFAULT_TAIL_EPOCHS, help="统计轮次")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help="图片保存目录")
    
    parser.add_argument("-a", "--all", action="store_true", help="绘制所有图片")
    parser.add_argument("--ablation", action="store_true", help="绘制消融实验分组柱状图")
    parser.add_argument("--sensitivity", action="store_true", help="绘制参数敏感性分析折线图")
    parser.add_argument("--mia-analysis", action="store_true", help="绘制MIA综合分析图（包含3个子图：Local Accuracy, Neighbor Accuracy, MIA Attack Success Rate）")
    
    args = parser.parse_args()
    
    # 如果设置了 -a/--all，则启用所有绘图选项
    if args.all:
        args.ablation = True
        args.sensitivity = True
        args.mia_analysis = True
    
    if not (args.ablation or args.sensitivity or args.mia_analysis):
        print("⚠️  未指定要绘制的图表，使用 --ablation 绘制消融实验图，或使用 --sensitivity 绘制敏感性分析图，或使用 --mia-analysis 绘制MIA综合分析图，或使用 -a/--all 绘制所有图片")
        args.mia_analysis = True
    
    if args.ablation:
        print("\n📊 正在绘制消融实验分组柱状图...")
        plot_ablation_study()
    
    if args.sensitivity:
        print("\n📊 正在绘制参数敏感性分析折线图...")
        SensitivityAnalysisPlotter.plot()
    
    if args.mia_analysis:
        print("\n📊 正在绘制MIA综合分析图...")
        plotter = MiaAnalysisPlotter(args.output_dir, args.tail_epochs, args.fig_dir)
        plotter.plot()
    
    if args.ablation or args.sensitivity or args.mia_analysis:
        print(f"\n✅ 所有图表已保存到: {args.fig_dir}")


if __name__ == "__main__":
    main()


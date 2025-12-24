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

class AbalationStudyPlotter:
    """
    消融实验分组柱状图绘制器
    
    绘制顶刊学术风格的消融实验分组柱状图。
    特点：Times New Roman字体、大字号、专业配色、纹理填充、去边框。
    """
    
    @staticmethod
    def _set_academic_style():
        """配置学术风格的绘图参数"""
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
    
    @staticmethod
    def _get_data():
        """返回封装好的实验数据"""
        return {
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
    
    @classmethod
    def plot(cls, save_name="ablation_study_comparison"):
        """
        生成消融实验分组柱状图
        
        Args:
            save_name: 保存文件名前缀
        """
        # ================= 0. 全局样式设置 (Academic Style) =================
        cls._set_academic_style()
        
        # ================= 1. 数据准备 =================
        data = cls._get_data()
        
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

        # 路径处理
        save_dir = Path("figures") # 或者是 DEFAULT_FIG_DIR
        save_dir.mkdir(exist_ok=True)

        # ================= 3. 循环绘图 - 为每个组合创建独立图片 =================
        for row_idx, dataset in enumerate(datasets):
            for col_idx, metric in enumerate(metrics):
                # 为每个组合创建独立的图片
                fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), dpi=300)
                
                # 数据提取
                y_data = data[dataset][metric]
                std_key = "Local Std" if metric == "Local Accuracy" else "Neighbor Std"
                y_err = data[dataset][std_key]
                
                # 绘制柱子
                for i, method in enumerate(methods):
                    offset = (i - 1.5) * width
                    
                    # 每个图片都设置图例 Label
                    label = method
                    
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
                # 坐标轴
                ax.set_xlabel(r"Privacy Budget ($\epsilon$)", fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(epsilon_labels)
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
                    ax.set_ylim(80, 102) 
                else:
                    ax.set_ylim(40, 75)

                # ================= 4. 图例与保存 =================
                # 获取图例句柄
                handles, labels = ax.get_legend_handles_labels()
                
                # 在每个图片的上侧居中放置图例，无边框，背景透明，2行2列布局
                ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
                         ncol=2, frameon=False, columnspacing=1.5)

                plt.tight_layout()
                # 调整顶部边距，防止标题被图例遮挡
                plt.subplots_adjust(top=0.85)

                # 生成文件名：基于数据集和指标
                dataset_short = dataset.replace("-", "_").replace(" ", "_").lower()
                metric_short = metric.replace(" ", "_").lower()
                pdf_path = save_dir / f"{save_name}_{dataset_short}_{metric_short}.pdf"
                
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
        # 使用区分度高的浅色系，便于区分不同隐私预算
        # 浅色系但对比度高的配色方案
        colors_A = ['#81c784', '#64b5f6', '#ba68c8', '#ffb74d']  # 浅绿 -> 浅蓝 -> 浅紫 -> 浅橙(无噪)
        colors_B = ['#81c784', '#64b5f6', '#ba68c8']  # 浅绿 -> 浅蓝 -> 浅紫

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
    def _darken_color(color, darken_factor=0.4):
        """将颜色变深，用于点的填充色"""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        # 向黑色方向混合（变深）
        darkened = tuple(c * (1 - darken_factor) for c in rgb)
        return darkened
    
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
            base_color = SensitivityAnalysisPlotter._adjust_color_for_neighbor(color) if is_neighbor else color
            # 线条颜色使用更深的版本，增加对比度
            line_color = SensitivityAnalysisPlotter._darken_color(base_color, darken_factor=0.3)
            # 增加线条宽度，使线条更明显
            line_width = 2.0  # Local 线条更粗，更突出
            # 点的填充色使用黑色，边缘色也是黑色（实心黑点）
            ax.plot(xs, [y_index]*len(xs), z_values, 
                    color=line_color, linewidth=line_width, linestyle=linestyle,
                    marker='o', markersize=6, 
                    markerfacecolor=line_color, markeredgecolor='white', markeredgewidth=0.5,
                    zorder=10 + y_index, label=label)
            
            # 2. 填充面 (PolyCollection) - Local 和 Neighbor 都绘制
            verts = []
            # 底部基准线 (z=zmin)
            z_min = zlim[0]
            polygon = [(x, z_min) for x in xs] + [(x, z) for x, z in zip(xs, z_values)][::-1]
            verts.append(polygon)
            
            # 填充面使用更浅的颜色和更高的透明度，以突出线条
            fill_color = base_color  # 使用基础颜色，而不是加深后的线条颜色
            # 降低填充面的透明度，使线条更明显
            fill_alpha = 0.2 if is_neighbor else 0.25
            poly = PolyCollection(verts, facecolors=fill_color, edgecolors='none', 
                                 alpha=fill_alpha, linewidths=0)
            ax.add_collection3d(poly, zs=y_index, zdir='y')
        
        # 获取 Local 和 Neighbor 数据
        loc_data_list = dataset_data[0]
        ngh_data_list = dataset_data[1]

        for i in range(num_eps):
            # 颜色：越靠前（epsilon 越小）颜色越深，或者反之
            c = colors[i]
            
            # 绘制 Local Ribbon（实线，带填充）
            add_ribbon(i, loc_data_list[i], c, label=None, 
                      linestyle='-', is_neighbor=False)
            
            # 绘制 Neighbor Line（虚线，无填充，使用稍浅的颜色）
            add_ribbon(i, ngh_data_list[i], c, label=None, 
                      linestyle='--', is_neighbor=True)

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
        
        # ax.grid(False) # 移除默认网格
        
        # 手动添加 Z 轴网格线 (仅在背板) - 使用更明显的颜色
        for z in np.linspace(zlim[0], zlim[1], 5):
            ax.plot([xs[0], xs[-1]], [num_eps-1, num_eps-1], [z, z], 
                    color='#999999', alpha=0.3, linestyle='--', linewidth=0.8)

    @classmethod
    def plot(cls, save_name="sensitivity_analysis", show_plot=True):
        """
        生成参数敏感性分析的3D Ribbon图
        
        Args:
            save_name: 保存文件名前缀
            show_plot: 是否显示图表（默认True，False则只保存）
        """
        cls._set_academic_style()
        data_pack = cls._get_data()
        
        # 为每个子图生成独立的图片
        for ds_conf in data_pack["datasets"]:
            ds_name = ds_conf["name"]
            indices = ds_conf["indices"] # [loc_idx, ngh_idx]
            zlim = ds_conf["zlim"]
            
            num_params = len(data_pack["params"])
            for i, param_conf in enumerate(data_pack["params"]):
                # 为每个子图创建独立的图片
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                
                # 提取该参数下，该数据集的 Local 和 Neighbor 数据
                current_ds_data = [param_conf["data"][indices[0]], param_conf["data"][indices[1]]]
                
                # 为每个图片单独计算z轴范围
                all_values = []
                for data_list in current_ds_data:
                    for data in data_list:
                        all_values.extend(data)
                z_min = min(all_values)
                z_max = max(all_values)
                # 添加边距（5%的边距）
                z_range = z_max - z_min
                z_margin = z_range * 0.05
                zlim = (z_min - z_margin, z_max + z_margin)
                
                # 每个图片都显示z轴标签
                show_zlabel = True
                
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
                
                # 为每个图片添加图例（1行，只有两个图例项：Local 和 Neighbor）
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Local'),
                    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Neighbor'),
                ]
                ax.legend(handles=legend_elements, loc='upper center', 
                         bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=14, 
                         frameon=False, columnspacing=1.0)

                # 手动调整边距，为3D图的轴标签和图例留出足够空间
                # 不使用 tight_layout，因为它对3D图支持不好
                plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
                
                # 生成文件名：基于数据集和参数
                ds_short = ds_name.lower().replace(' ', '_')
                # 从title中提取参数名称，例如 "(a) Impact of Rank" -> "rank"
                title_lower = param_conf["title"].lower()
                if "rank" in title_lower:
                    param_short = "rank"
                elif "topk" in title_lower or "topm" in title_lower:
                    param_short = "topk"
                elif "schedule" in title_lower or "factor" in title_lower or "p" in title_lower:
                    param_short = "p"
                else:
                    param_short = title_lower.replace(' ', '_').replace('(', '').replace(')', '').replace('$', '').replace('impact_of_', '').replace('_', '')
                save_path = Path("figures") / f"{save_name}_{ds_short}_{param_short}.pdf"
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
        
        # 图例（放在左下角，2x2格式）
        if show_legend:
            ax.legend(loc='lower left', frameon=False, ncol=2)
    
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
        # 降低最小值以为图例留出空间
        combined_accs = all_local_accs + all_neighbor_accs
        if combined_accs:
            min_acc = min(combined_accs)
            max_acc = max(combined_accs)
            # 降低最小值，为左下角图例留出空间
            shared_y_lim = (max(0, min_acc - 30), min(105, max_acc + 10))
        else:
            shared_y_lim = None
        
        # 计算 MIA 的Y轴范围
        # 降低最小值以为图例留出空间
        if all_acc_accs:
            min_mia = min(all_acc_accs)
            max_mia = max(all_acc_accs)
            # 降低最小值，为左下角图例留出空间
            mia_y_lim = (max(0, min_mia - 30), min(105, max_mia + 10))
        else:
            mia_y_lim = None
        
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 统一的绘图区域布局参数（确保三张图的绘图区域大小一致）
        plot_left = 0.12
        plot_right = 0.95
        plot_bottom = 0.12
        plot_top = 0.95
        
        # 绘制第一张图：Local Accuracy
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        self._plot_subplot(
            ax1, 
            dataset_local_map, 
            datasets, 
            noise_list,
            'Accuracy (%)',
            title=None,
            show_legend=True,
            y_lim=shared_y_lim
        )
        plt.subplots_adjust(left=plot_left, right=plot_right, bottom=plot_bottom, top=plot_top)
        output_path1 = self.fig_dir / 'mia_local_accuracy.pdf'
        plt.savefig(output_path1, bbox_inches='tight', dpi=300)
        print(f"✅ Local Accuracy图已保存: {output_path1}")
        plt.close()
        
        # 绘制第二张图：Neighbor Accuracy
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        self._plot_subplot(
            ax2, 
            dataset_neighbor_map, 
            datasets, 
            noise_list,
            'Accuracy (%)',
            title=None,
            show_legend=True,
            y_lim=shared_y_lim
        )
        plt.subplots_adjust(left=plot_left, right=plot_right, bottom=plot_bottom, top=plot_top)
        output_path2 = self.fig_dir / 'mia_neighbor_accuracy.pdf'
        plt.savefig(output_path2, bbox_inches='tight', dpi=300)
        print(f"✅ Neighbor Accuracy图已保存: {output_path2}")
        plt.close()
        
        # 绘制第三张图：MIA Success Rate（包含图例和baseline）
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        self._plot_subplot(
            ax3, 
            dataset_mia_map, 
            datasets, 
            noise_list,
            'Accuracy (%)',
            title=None,
            show_legend=True,
            y_lim=mia_y_lim
        )
        
        # 在第三个子图上绘制50%基线虚线
        baseline_value = 50.0
        ax3.axhline(y=baseline_value, color='gray', linestyle='--', linewidth=5, 
                   alpha=0.7, zorder=5, label='_nolegend_')  # _nolegend_ 确保不显示在图例中
        
        # 获取当前Y轴范围，用于确定箭头位置
        y_min, y_max = ax3.get_ylim()
        x_min, x_max = ax3.get_xlim()
        
        # 计算箭头起始位置（在图的左侧，稍微高于50%基线）
        arrow_x_start = x_min + (x_max - x_min) * 0.65  # 距离左边界65%的位置
        arrow_y_start = baseline_value + (y_max - baseline_value) * 0.15  # 基线以上15%的位置
        
        # 箭头指向的位置（在50%基线上）
        arrow_x_end = x_min + (x_max - x_min) * 0.3  # 距离左边界30%的位置
        arrow_y_end = baseline_value
        
        # 绘制箭头和文字标注
        ax3.annotate('baseline', 
                    xy=(arrow_x_end, arrow_y_end),  # 箭头指向的位置
                    xytext=(arrow_x_start, arrow_y_start),  # 文字位置
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.7),
                    fontsize=plt.rcParams['legend.fontsize'],
                    color='gray',
                    ha='left',
                    va='bottom',
                    zorder=15)
        
        # 第三张图也使用和前两张图相同的布局参数（图例在左下角，不需要额外空间）
        plt.subplots_adjust(left=plot_left, right=plot_right, bottom=plot_bottom, top=plot_top)
        output_path3 = self.fig_dir / 'mia_success_rate.pdf'
        plt.savefig(output_path3, bbox_inches='tight', dpi=300)
        print(f"✅ MIA Success Rate图已保存: {output_path3}")
        plt.close()


# ================= 梯度聚类可视化绘图类 =================
class GradientClusteringPlotter:
    """梯度聚类 t-SNE 可视化绘图类"""
    
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR, 
                 fig_dir: Path = DEFAULT_FIG_DIR):
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
            'mathtext.fontset': 'stix',
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'axes.linewidth': 1.5,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2,
            'lines.markersize': 8,
        })
    
    def _load_clustering_data(self, exp_name: str, dataset: str, epoch: int = 40, 
                             config: Optional[Dict[str, Any]] = None, noise: Optional[float] = None):
        """加载梯度聚类数据文件（仅精确匹配）"""
        if config is None:
            print(f"⚠️  警告: 未提供配置信息，无法进行精确匹配")
            return None
        
        base_dir = self.output_dir / exp_name / dataset
        factorization = config.get('factorization_list', ['sepfpl'])[0]
        rank = config.get('rank_list', [8])[0]
        noise_value = noise if noise is not None else config.get('noise_list', [0])[0]
        seed = config.get('seed_list', [1])[0]
        num_users = config.get('num_users_list', [50])[0]
        noise_str = str(noise_value)
        
        if factorization in ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
            topk = config.get('sepfpl_topk', 25)
            rdp_p = config.get('rdp_p', 0.2)
            filename_pattern = f'gc_e{epoch}_sepfpl_{rank}_{noise_str}_{seed}_{topk}_{str(rdp_p)}_{num_users}.pkl'
        else:
            filename_pattern = f'gc_e{epoch}_{factorization}_{rank}_{noise_str}_{seed}_{num_users}.pkl'
        
        data_file = base_dir / filename_pattern
        if not data_file.exists():
            print(f"⚠️  警告: 数据文件不存在: {data_file}")
            return None
        
        try:
            with open(data_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ 错误: 无法读取数据文件 {data_file}: {e}")
            return None
    
    def plot(self, exp_name: str = "exp5-gradient-clustering", epoch: Optional[int] = None):
        """生成梯度聚类 t-SNE 可视化图"""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("❌ 错误: 需要安装 scikit-learn 库: pip install scikit-learn")
            return
        
        if 'EXPERIMENT_5_GRADIENT_CLUSTERING' not in EXPERIMENT_CONFIGS:
            print("❌ 错误: 找不到 EXPERIMENT_5_GRADIENT_CLUSTERING 配置")
            return
        
        config = EXPERIMENT_CONFIGS['EXPERIMENT_5_GRADIENT_CLUSTERING']
        datasets = config.get('dataset_list', [])
        total_rounds = config.get('round', 10)
        noise_list = config.get('noise_list', [0.0])
        # 每2个epoch绘制一次（2, 4, 6, 8, 10...）
        epochs_to_plot = [10]
        
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
        # CIFAR-100 粗类别映射（100个细类别 -> 20个粗类别）
        cifar100_coarse_labels = np.array([
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ])

        food101_coarse_labels = np.array([
            0, 1, 0, 1, 1, 5, 0, 3, 0, 4, 6, 5, 0, 5, 0, 2, 6, 0, 1, 1,
            0, 0, 0, 7, 4, 2, 0, 4, 0, 6, 0, 3, 5, 7, 6, 6, 1, 6, 1, 6,
            7, 7, 2, 3, 0, 6, 3, 5, 4, 2, 5, 3, 4, 7, 4, 6, 0, 3, 2, 4,
            3, 0, 7, 2, 6, 7, 6, 2, 3, 3, 0, 0, 1, 3, 3, 1, 6, 1, 4, 3,
            3, 0, 3, 6, 2, 2, 5, 2, 3, 3, 5, 1, 0, 2, 4, 6, 0, 2, 0, 7, 7
        ])

        # Mapping from 120 Stanford Dogs classes to 8 visual coarse labels
        # Indices correspond to the standard alphabetical sorting of n0... folder names
        stanford_dogs_coarse_labels = np.array([
            # 0-9: Chihuahua to Afghan
            0, 4, 3, 3, 3, 4, 0, 0, 2, 3, 
            # 10-19: Basset to Irish Wolfhound
            0, 0, 5, 5, 2, 0, 0, 2, 3, 5, 
            # 20-29: Italian Greyhound to Am. Staffordshire
            2, 2, 2, 1, 5, 3, 5, 2, 0, 0, 
            # 30-39: Bedlington to Sealyham
            7, 5, 7, 5, 4, 4, 3, 5, 4, 6, 
            # 40-49: Airedale to Tibetan Terrier
            5, 5, 4, 4, 0, 6, 7, 7, 5, 7, 
            # 50-59: Silky to German Short-haired
            3, 7, 6, 3, 1, 7, 5, 2, 2, 2, 
            # 60-69: Vizsla to Sussex
            2, 6, 5, 5, 6, 6, 6, 6, 5, 4, 
            # 70-79: Irish Water to Shetland
            7, 6, 0, 1, 2, 7, 2, 7, 6, 1, 
            # 80-89: Collie to Appenzeller
            1, 6, 7, 2, 2, 2, 0, 2, 1, 2, 
            # 90-99: Entlebucher to Siberian Husky
            2, 0, 0, 1, 0, 2, 1, 1, 1, 1, 
            # 100-109: Affenpinscher to Keeshond
            0, 0, 0, 1, 1, 6, 6, 1, 1, 1, 
            # 110-119: Brabancon to African Hunting Dog
            0, 0, 0, 7, 7, 7, 2, 2, 2, 6 
        ])

        # Label Names for Reference
        coarse_label_names = {
            0: "Small_Smooth_Common",
            1: "Giant_Fluffy",
            2: "Athletic_Medium",
            3: "Long_Silky",
            4: "Spaniels_Small",
            5: "Hounds_Rough",
            6: "White_Spotted",
            7: "Curly_Textured"
        }
        
        # 未填充标记（不支持 edgecolors）
        unfilled_markers = {'1', '2', '3', '4', '+', 'x', '|', '_', '.', ',', '0'}
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', 'd', 'H', 
                  '8', '1', '2', '3', '4', '+', 'x', '|', '_', '.', ',', '1', '2', '3', '4']
        
        for dataset in datasets:
            for noise in noise_list:
                for epoch in epochs_to_plot:
                    print(f"\n📊 正在处理数据集 {dataset}, Noise {noise}, Epoch {epoch}...")
                    data = self._load_clustering_data(exp_name, dataset, epoch, config=config, noise=noise)
                    if data is None:
                        print(f"⚠️  警告: 数据集 {dataset} (Noise {noise}) 在 Epoch {epoch} 没有数据，跳过")
                        continue
                    
                    gradient_vectors = np.array(data['gradient_vectors'])
                    all_client_labels = np.array(data['client_labels'])
                    all_community_ids = np.array(data['community_ids'])
                    client_ids = np.array(data['client_ids'])
                    # 获取类别名称（如果存在）
                    all_client_classnames = data.get('client_classnames', [])
                    
                    if len(gradient_vectors) == 0:
                        print(f"⚠️  警告: 数据集 {dataset} (Epoch {epoch}) 没有梯度数据")
                        continue
                    
                    # L2 归一化
                    norms = np.linalg.norm(gradient_vectors, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    gradient_vectors = gradient_vectors / norms
                    
                    # 获取对应的标签和社区ID
                    client_labels = all_client_labels[client_ids]
                    community_ids = all_community_ids[client_ids]
                    
                    # 细类别映射到粗类别（仅对 CIFAR-100）
                    if dataset.lower() in ['cifar-100', 'cifar100', 'cifar_100']:
                        client_labels = np.array([
                            cifar100_coarse_labels[label] if 0 <= label < len(cifar100_coarse_labels) else label
                            for label in client_labels
                        ])
                    # food101 和 Stanford Dogs 不需要粗粒度映射，保持原始标签
                    
                    # t-SNE 降维
                    print(f"📊 正在对 {dataset} (Epoch {epoch}) 进行 t-SNE 降维...")
                    n_samples = len(gradient_vectors)
                    max_perplexity = min(50, max(5, (n_samples - 1) // 3))
                    perplexity = min(30, max_perplexity)
                    
                    tsne = TSNE(
                        n_components=2,
                        random_state=42,
                        perplexity=perplexity,
                        max_iter=1000,
                        learning_rate=200,
                        early_exaggeration=12,
                        min_grad_norm=1e-7,
                        metric='euclidean',
                        init='random'
                    )
                    embeddings = tsne.fit_transform(gradient_vectors)
                    
                    # 创建图形
                    fig, ax = plt.subplots(figsize=(6, 6))
                    unique_labels = np.unique(client_labels)
                    unique_communities = np.unique(community_ids)
                    
                    # 分配颜色给社区（Cluster）- 根据社区数量选择不同的颜色方案
                    n_communities = len(unique_communities)
                    if n_communities <= 10:
                        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_communities]
                    elif n_communities <= 20:
                        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_communities]
                    else:
                        colors = plt.cm.tab20(np.linspace(0, 1, n_communities))
                    
                    community_to_color = {comm: colors[i] for i, comm in enumerate(sorted(unique_communities))}
                    
                    # 对于 food101 和 Stanford Dogs，不使用 marker 区分 label，只用颜色区分 Cluster
                    # 对于 CIFAR-100，使用 marker 区分粗粒度标签
                    use_marker_for_labels = dataset.lower() in ['cifar-100', 'cifar100', 'cifar_100']
                    
                    if use_marker_for_labels:
                        # 分配标记给粗粒度标签（仅 CIFAR-100）
                        label_to_marker = {label: markers[i % len(markers)] for i, label in enumerate(sorted(unique_labels))}
                    
                    # 绘制散点图
                    for i in range(len(embeddings)):
                        color = community_to_color[community_ids[i]]  # 颜色表示社区
                        if use_marker_for_labels:
                            marker = label_to_marker[client_labels[i]]   # 标记表示粗粒度标签（仅 CIFAR-100）
                            scatter_kwargs = {'c': [color], 'marker': marker, 's': 100, 'alpha': 0.7}
                            if marker not in unfilled_markers:
                                scatter_kwargs.update({'edgecolors': 'black', 'linewidths': 1.5})
                        else:
                            # food101 和 Stanford Dogs：统一使用圆形标记
                            scatter_kwargs = {'c': [color], 'marker': 'o', 's': 100, 'alpha': 0.7, 
                                            'edgecolors': 'black', 'linewidths': 1.5}
                        ax.scatter(embeddings[i, 0], embeddings[i, 1], **scatter_kwargs)
                    
                    # 暂时不需要图例
                    # # 添加图例
                    # from matplotlib.patches import Patch
                    # 
                    # # 社区图例（颜色）
                    # community_legend_elements = []
                    # for comm in sorted(unique_communities):
                    #     community_legend_elements.append(
                    #         Patch(facecolor=community_to_color[comm], label=f'Cluster {comm}', edgecolor='black', linewidth=1)
                    #     )
                    # 
                    # # 粗粒度类别图例（标记）- 仅对 CIFAR-100
                    # label_legend_elements = []
                    # if use_marker_for_labels:
                    #     for label in sorted(unique_labels):
                    #         marker = label_to_marker[label]
                    #         label_legend_elements.append(
                    #             plt.Line2D([0], [0], marker=marker, color='w', label=f'Group {label}',
                    #                      markerfacecolor='gray', markeredgecolor='black', markersize=10, linestyle='None')
                    #         )
                    
                    # # 设置标签和标题
                    # ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=14)
                    # ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=14)
                    # dataset_display_name = {
                    #     'stanford_dogs': 'Stanford Dogs',
                    #     'food-101': 'Food-101',
                    #     'food101': 'Food-101',
                    #     'food_101': 'Food-101'
                    # }.get(dataset.lower(), dataset)
                    # ax.set_title(f'{dataset_display_name} - Gradient Clustering (Epoch {epoch}, Noise={noise})', 
                    #            fontweight='bold', pad=15, fontsize=16)
                    
                    # 美化
                    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_linewidth(1.5)
                    ax.spines['bottom'].set_linewidth(1.5)
                    
                    # # 创建图例并放在图片外
                    # has_community_legend = len(community_legend_elements) > 0
                    # has_label_legend = use_marker_for_labels and len(label_legend_elements) > 0 and len(unique_labels) <= 15
                    # 
                    # if has_community_legend and has_label_legend:
                    #     # 两个图例：左侧和右侧（仅 CIFAR-100）
                    #     legend1 = fig.legend(handles=community_legend_elements, 
                    #                        loc='center left', bbox_to_anchor=(0, 0.5),
                    #                        title='Clusters', frameon=True, 
                    #                        fancybox=True, shadow=True, fontsize=10, title_fontsize=11)
                    #     legend2 = fig.legend(handles=label_legend_elements, 
                    #                        loc='center right', bbox_to_anchor=(1, 0.5),
                    #                        title='Coarse Groups', frameon=True,
                    #                        fancybox=True, shadow=True, fontsize=10, title_fontsize=11, ncol=1)
                    #     # 调整布局为图例留出空间
                    #     plt.subplots_adjust(left=0.15, right=0.85)
                    # elif has_community_legend:
                    #     # 只有一个图例：放在右侧（food101 和 Stanford Dogs）
                    #     legend1 = fig.legend(handles=community_legend_elements, 
                    #                        loc='center right', bbox_to_anchor=(1, 0.5),
                    #                        title='Clusters', frameon=True, 
                    #                        fancybox=True, shadow=True, fontsize=10, title_fontsize=11)
                    #     plt.subplots_adjust(right=0.85)
                    
                    plt.tight_layout()
                    
                    # 保存图片
                    dataset_short = dataset.replace('-', '_').replace(' ', '_').lower()
                    factorization = config.get('factorization_list', ['sepfpl'])[0]
                    rank = config.get('rank_list', [8])[0]
                    seed = config.get('seed_list', [1])[0]
                    num_users = config.get('num_users_list', [50])[0]
                    noise_str = str(noise)
                    
                    if factorization in ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
                        topk = config.get('sepfpl_topk', 25)
                        rdp_p = config.get('rdp_p', 0.2)
                        filename = f'gc_{dataset_short}_e{epoch}_{factorization}_{rank}_{noise_str}_{seed}_{topk}_{str(rdp_p)}_{num_users}.pdf'
                    else:
                        filename = f'gc_{dataset_short}_e{epoch}_{factorization}_{rank}_{noise_str}_{seed}_{num_users}.pdf'
                    
                    save_path = self.fig_dir / filename
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    print(f"✅ 梯度聚类可视化图已保存: {save_path}")
                    plt.close()


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验结果可视化工具")
    
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="数据目录")
    parser.add_argument("--tail-epochs", type=int, default=DEFAULT_TAIL_EPOCHS, help="统计轮次")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help="图片保存目录")
    
    parser.add_argument("-a", "--all", action="store_true", help="绘制所有图片")
    parser.add_argument("-b", "--ablation", action="store_true", help="绘制消融实验分组柱状图")
    parser.add_argument("-s", "--sensitivity", action="store_true", help="绘制参数敏感性分析折线图")
    parser.add_argument("-m", "--mia-analysis", action="store_true", help="绘制MIA综合分析图（包含3个子图：Local Accuracy, Neighbor Accuracy, MIA Attack Success Rate）")
    parser.add_argument("-g", "--gradient-clustering", action="store_true", help="绘制梯度聚类可视化图（t-SNE降维）")
    
    args = parser.parse_args()
    
    # 如果设置了 -a/--all，则启用所有绘图选项
    if args.all:
        args.ablation = True
        args.sensitivity = True
        args.mia_analysis = True
        args.gradient_clustering = True
    
    if not (args.ablation or args.sensitivity or args.mia_analysis or args.gradient_clustering):
        print("⚠️  未指定要绘制的图表，使用 --ablation 绘制消融实验图，或使用 --sensitivity 绘制敏感性分析图，或使用 --mia-analysis 绘制MIA综合分析图，或使用 --gradient-clustering 绘制梯度聚类可视化图，或使用 -a/--all 绘制所有图片")
        args.mia_analysis = True
    
    if args.ablation:
        print("\n📊 正在绘制消融实验分组柱状图...")
        AbalationStudyPlotter.plot()
    
    if args.sensitivity:
        print("\n📊 正在绘制参数敏感性分析折线图...")
        SensitivityAnalysisPlotter.plot()
    
    if args.mia_analysis:
        print("\n📊 正在绘制MIA综合分析图...")
        plotter = MiaAnalysisPlotter(args.output_dir, args.tail_epochs, args.fig_dir)
        plotter.plot()
    
    if args.gradient_clustering:
        print("\n📊 正在绘制梯度聚类可视化图...")
        plotter = GradientClusteringPlotter(args.output_dir, args.fig_dir)
        plotter.plot()
    
    if args.ablation or args.sensitivity or args.mia_analysis or args.gradient_clustering:
        print(f"\n✅ 所有图表已保存到: {args.fig_dir}")


if __name__ == "__main__":
    main()


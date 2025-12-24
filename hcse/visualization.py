#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码树和结构熵可视化模块

提供编码树和结构熵的多种可视化方式：
- 编码树结构图：展示编码树的层次结构和节点关系
- 原始图结构：可视化原始图的网络结构
- 结构熵分布热力图：显示各节点的结构熵值分布
- 交互式可视化：基于Plotly的交互式多视图展示

=== 调用关系图 ===

模块级函数调用关系:
├── get_chinese_font() -> str
├── create_visualization_demo() -> Tuple[EncodingTreeVisualizer, str]
│   ├── generate_random_undirected_graph() [from encoding_tree]
│   ├── PartitionTree() [from encoding_tree]
│   ├── tree.build_encoding_tree()
│   ├── EncodingTreeVisualizer()
│   └── visualizer.create_comprehensive_report()
└── create_advanced_visualization_demo() -> Tuple[EncodingTreeVisualizer, str]
    ├── generate_connected_random_graph() [from encoding_tree]
    ├── PartitionTree() [from encoding_tree]
    ├── tree.build_encoding_tree()
    ├── EncodingTreeVisualizer()
    └── visualizer.create_comprehensive_report()

EncodingTreeVisualizer 类方法调用关系:
├── __init__(partition_tree, adj_matrix)
├── visualize_tree_structure() -> Optional[plt.Figure]
│   ├── _calculate_node_positions()
│   ├── _draw_connections()
│   ├── _draw_nodes()
│   └── _add_legend()
├── visualize_original_graph() -> Optional[plt.Figure]
├── visualize_entropy_heatmap() -> Optional[plt.Figure]
├── create_interactive_visualization() -> Optional[go.Figure]
│   ├── _add_tree_to_plotly()
│   │   └── _calculate_node_positions()
│   ├── _add_graph_to_plotly()
│   ├── _add_entropy_to_plotly()
│   └── _add_statistics_to_plotly()
└── create_comprehensive_report() -> str
    ├── visualize_tree_structure()
    ├── visualize_original_graph()
    ├── visualize_entropy_heatmap()
    └── create_interactive_visualization()

主要调用流程:
1. 演示函数 -> 生成测试图 -> 构建编码树 -> 创建可视化器 -> 生成报告
2. 可视化器方法 -> 内部辅助方法 -> matplotlib/plotly绘图
3. 交互式可视化 -> 多个子图绘制方法 -> 综合展示

依赖模块：
- encoding_tree：编码树构建和计算
- matplotlib：静态图表绘制
- plotly：交互式可视化
- networkx：图结构处理
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import os
import warnings
warnings.filterwarnings('ignore')

# 导入优化后的编码树模块
from .encoding_tree import (
    PartitionTree, 
    PartitionTreeNode, 
    generate_random_undirected_graph,
    generate_connected_random_graph,
    NodeId,
    AdjacencyMatrix,
    NodeDict
)

# 设置中文字体
import matplotlib.font_manager as fm

# 项目根目录和输出路径配置
PROJECT_ROOT = "/home/liuxin25/code/DP-FPL/hcse"
VISUALIZATION_OUTPUT_DIR = f"{PROJECT_ROOT}/visualization_output"
ADVANCED_VISUALIZATION_OUTPUT_DIR = f"{PROJECT_ROOT}/advanced_visualization_output"


def get_chinese_font() -> str:
    """获取系统中可用的中文字体。
    
    按优先级顺序检测系统中可用的中文字体，支持文泉驿、思源、微软等常见字体。
    如果找不到中文字体，则返回默认的DejaVu Sans字体。
    
    Returns:
        str: 可用的中文字体名称
    """
    font_list = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [
        'WenQuanYi Micro Hei',  # 文泉驿微米黑
        'WenQuanYi Zen Hei',    # 文泉驿正黑
        'Noto Sans CJK SC',     # 思源黑体
        'Noto Serif CJK SC',    # 思源宋体
        'Source Han Sans SC',   # 思源黑体
        'Source Han Serif SC',  # 思源宋体
        'SimHei',               # 黑体
        'SimSun',               # 宋体
        'Microsoft YaHei',      # 微软雅黑
        'DejaVu Sans'           # 备用字体
    ]
    
    for font in chinese_fonts:
        if font in font_list:
            return font
    
    return 'DejaVu Sans'  # 默认备用字体


# 设置字体
chinese_font = get_chinese_font()
plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print(f"使用字体: {chinese_font}")


class EncodingTreeVisualizer:
    """编码树可视化器
    
    提供编码树和结构熵的多种可视化方式，包括静态图表和交互式可视化。
    支持编码树结构图、原始图结构、结构熵分布热力图等可视化功能。
    
    方法调用关系:
    ├── __init__() - 初始化可视化器
    ├── visualize_tree_structure() - 编码树结构可视化
    │   ├── _calculate_node_positions() - 计算节点位置
    │   ├── _draw_connections() - 绘制连接线
    │   ├── _draw_nodes() - 绘制节点
    │   └── _add_legend() - 添加图例
    ├── visualize_original_graph() - 原始图结构可视化
    ├── visualize_entropy_heatmap() - 结构熵热力图可视化
    ├── create_interactive_visualization() - 交互式可视化
    │   ├── _add_tree_to_plotly() - 添加编码树到Plotly
    │   │   └── _calculate_node_positions() - 计算节点位置
    │   ├── _add_graph_to_plotly() - 添加原始图到Plotly
    │   ├── _add_entropy_to_plotly() - 添加熵分布到Plotly
    │   └── _add_statistics_to_plotly() - 添加统计信息到Plotly
    └── create_comprehensive_report() - 生成综合报告
        ├── visualize_tree_structure()
        ├── visualize_original_graph()
        ├── visualize_entropy_heatmap()
        └── create_interactive_visualization()
    """
    
    def __init__(self, partition_tree: PartitionTree, adj_matrix: Optional[AdjacencyMatrix] = None):
        """初始化可视化器。
        
        Args:
            partition_tree: PartitionTree对象，包含编码树结构信息
            adj_matrix: 原始图的邻接矩阵，用于原始图结构可视化
        """
        self.tree = partition_tree
        self.adj_matrix = adj_matrix
        self.node_positions: Dict[NodeId, Tuple[float, float]] = {}
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
    def visualize_tree_structure(self, figsize: Tuple[int, int] = (15, 10), 
                               save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """使用matplotlib可视化编码树结构。
        
        创建编码树的层次结构可视化图，显示节点间的父子关系。
        使用不同颜色和形状区分根节点、内部节点和叶子节点。
        
        调用关系:
        ├── _calculate_node_positions() - 计算节点位置
        ├── _draw_connections() - 绘制连接线
        ├── _draw_nodes() - 绘制节点
        └── _add_legend() - 添加图例
        
        Args:
            figsize: 图像大小，默认(15, 10)
            save_path: 保存路径，None表示不保存
        
        Returns:
            Optional[plt.Figure]: matplotlib图形对象，如果出错返回None
        """
        if not hasattr(self.tree, 'root_id') or self.tree.root_id is None:
            print("错误：编码树尚未构建")
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        
        # 计算节点位置
        self._calculate_node_positions()
        
        # 绘制连接线
        self._draw_connections(ax)
        
        # 绘制节点
        self._draw_nodes(ax)
        
        # 设置图形属性
        ax.set_title('Encoding Tree Structure', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # 添加图例
        self._add_legend(ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def _calculate_node_positions(self) -> None:
        """计算节点在图形中的位置。
        
        按层次组织编码树节点，为每个节点分配合适的坐标位置。
        使用递归方式遍历树结构，按层分配x坐标，按深度分配y坐标。
        """
        # 按层次组织节点
        levels: Dict[int, List[Tuple[NodeId, PartitionTreeNode]]] = {}
        
        def organize_by_level(node_id: NodeId, level: int) -> None:
            if node_id not in self.tree.tree_node:
                return
            
            if level not in levels:
                levels[level] = []
            
            node = self.tree.tree_node[node_id]
            levels[level].append((node_id, node))
            
            if node.children:
                for child_id in node.children:
                    organize_by_level(child_id, level + 1)
        
        organize_by_level(self.tree.root_id, 0)
        
        # 为每一层分配位置
        max_width = max(len(nodes) for nodes in levels.values()) if levels else 1
        
        for level, nodes in levels.items():
            y = -level * 2  # 从上到下排列
            x_positions = np.linspace(-max_width/2, max_width/2, len(nodes))
            
            for i, (node_id, node) in enumerate(nodes):
                self.node_positions[node_id] = (x_positions[i], y)
    
    def _draw_connections(self, ax: plt.Axes) -> None:
        """绘制节点之间的连接线。
        
        在matplotlib轴上绘制编码树节点间的父子连接线。
        
        Args:
            ax: matplotlib轴对象，用于绘制连接线
        """
        for node_id, (x, y) in self.node_positions.items():
            node = self.tree.tree_node[node_id]
            if node.children:
                for child_id in node.children:
                    if child_id in self.node_positions:
                        child_x, child_y = self.node_positions[child_id]
                        ax.plot([x, child_x], [y, child_y], 'k-', alpha=0.6, linewidth=1.5)
    
    def _draw_nodes(self, ax: plt.Axes) -> None:
        """绘制节点。
        
        在matplotlib轴上绘制编码树节点，使用不同颜色和形状区分节点类型。
        根节点用红色圆形表示，叶子节点用青色方形表示，内部节点用蓝色圆形表示。
        
        Args:
            ax: matplotlib轴对象，用于绘制节点
        """
        for node_id, (x, y) in self.node_positions.items():
            node = self.tree.tree_node[node_id]
            
            # 根据节点类型选择颜色和形状
            if node_id == self.tree.root_id:
                color = '#FF6B6B'  # 红色表示根节点
                size = 200
                shape = 'circle'
            elif not node.children or len(node.children) == 0:
                color = '#4ECDC4'  # 青色表示叶子节点
                size = 100
                shape = 'square'
            else:
                color = '#45B7D1'  # 蓝色表示内部节点
                size = 150
                shape = 'circle'
            
            # 绘制节点
            if shape == 'circle':
                circle = Circle((x, y), size/1000, facecolor=color, 
                              edgecolor='black', linewidth=2, alpha=0.8)
                ax.add_patch(circle)
            else:
                rect = FancyBboxPatch((x-size/2000, y-size/2000), size/1000, size/1000,
                                    boxstyle="round,pad=0.01", facecolor=color,
                                    edgecolor='black', linewidth=2, alpha=0.8)
                ax.add_patch(rect)
            
            # 添加节点标签
            if not node.children or len(node.children) == 0:
                label = f"Node {node.partition[0]}"
            else:
                partition_str = ','.join(map(str, node.partition[:3]))
                if len(node.partition) > 3:
                    partition_str += '...'
                label = f"ID:{node_id}\n[{partition_str}]"
            
            ax.text(x, y-0.3, label, ha='center', va='top', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _add_legend(self, ax: plt.Axes) -> None:
        """添加图例。
        
        为编码树可视化图添加图例，说明不同颜色和形状代表的节点类型。
        
        Args:
            ax: matplotlib轴对象，用于添加图例
        """
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                      markersize=10, label='Root Node'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4ECDC4', 
                      markersize=8, label='Leaf Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', 
                      markersize=9, label='Internal Node')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def visualize_original_graph(self, figsize: Tuple[int, int] = (12, 8), 
                               layout: str = 'spring', 
                               save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """可视化原始图结构。
        
        使用NetworkX和matplotlib可视化原始图结构，显示节点和边的连接关系。
        支持多种布局算法：spring（弹簧布局）、circular（圆形布局）、hierarchical（层次布局）。
        
        Args:
            figsize: 图像大小，默认(12, 8)
            layout: 布局算法，可选'spring', 'circular', 'hierarchical'
            save_path: 保存路径，None表示不保存
        
        Returns:
            Optional[plt.Figure]: matplotlib图形对象，如果出错返回None
        """
        if self.adj_matrix is None:
            print("错误：未提供邻接矩阵")
            return None
        
        # 创建NetworkX图
        G = nx.from_numpy_array(self.adj_matrix)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 选择布局算法
        layout_functions = {
            'spring': lambda: nx.spring_layout(G, k=1, iterations=50),
            'circular': lambda: nx.circular_layout(G),
            'hierarchical': lambda: nx.shell_layout(G)
        }
        
        pos = layout_functions.get(layout, layout_functions['spring'])()
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=[w/2 for w in weights], 
                              edge_color='gray')
        
        # 绘制节点
        node_colors = [self.colors[i % len(self.colors)] for i in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, alpha=0.8)
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # 添加边权重标签
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        ax.set_title('原始图结构', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def visualize_entropy_heatmap(self, figsize: Tuple[int, int] = (10, 8), 
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """可视化结构熵分布热力图。
        
        计算并可视化编码树中每个节点的结构熵值，以热力图形式展示。
        使用YlOrRd颜色映射，熵值越高颜色越深。
        
        Args:
            figsize: 图像大小，默认(10, 8)
            save_path: 保存路径，None表示不保存
        
        Returns:
            Optional[plt.Figure]: matplotlib图形对象，如果出错返回None
        """
        if not hasattr(self.tree, 'root_id') or self.tree.root_id is None:
            print("错误：编码树尚未构建")
            return None
        
        # 计算每个节点的结构熵
        entropy_data: List[float] = []
        node_ids: List[str] = []
        
        for node_id, node in self.tree.tree_node.items():
            if node.parent is not None:
                # 计算结构熵
                node_vol = node.vol
                node_g = node.g
                parent_vol = self.tree.tree_node[node.parent].vol
                
                if node_vol > 0 and parent_vol > 0:
                    entropy = -(node_g / self.tree.VOL) * math.log2(node_vol / parent_vol)
                else:
                    entropy = 0.0
                
                entropy_data.append(entropy)
                node_ids.append(f"节点{node_id}")
        
        # 创建热力图数据矩阵
        n_nodes = len(entropy_data)
        if n_nodes == 0:
            print("没有可用的熵数据")
            return None
        
        # 将熵数据组织成矩阵形式
        matrix_size = int(math.ceil(math.sqrt(n_nodes)))
        entropy_matrix = np.zeros((matrix_size, matrix_size))
        
        for i, entropy in enumerate(entropy_data):
            row = i // matrix_size
            col = i % matrix_size
            entropy_matrix[row, col] = entropy
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(entropy_matrix, cmap='YlOrRd', aspect='auto')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('结构熵值', rotation=270, labelpad=20)
        
        # 设置标签
        ax.set_title('结构熵分布热力图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('节点索引 (列)')
        ax.set_ylabel('节点索引 (行)')
        
        # 添加数值标注
        for i in range(matrix_size):
            for j in range(matrix_size):
                if entropy_matrix[i, j] != 0:
                    ax.text(j, i, f'{entropy_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def create_interactive_visualization(self, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """创建交互式可视化。
        
        使用Plotly创建包含四个子图的交互式可视化界面。
        包括编码树结构、原始图结构、结构熵分布和节点统计四个部分。
        
        调用关系:
        ├── _add_tree_to_plotly() - 添加编码树到Plotly子图
        ├── _add_graph_to_plotly() - 添加原始图到Plotly子图
        ├── _add_entropy_to_plotly() - 添加熵分布到Plotly子图
        └── _add_statistics_to_plotly() - 添加统计信息到Plotly子图
        
        Args:
            save_path: 保存路径，None表示不保存
        
        Returns:
            Optional[go.Figure]: Plotly图形对象，如果出错返回None
        """
        if not hasattr(self.tree, 'root_id') or self.tree.root_id is None:
            print("错误：编码树尚未构建")
            return None
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('编码树结构', '原始图结构', '结构熵分布', '节点统计'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. 编码树结构
        self._add_tree_to_plotly(fig, row=1, col=1)
        
        # 2. 原始图结构
        if self.adj_matrix is not None:
            self._add_graph_to_plotly(fig, row=1, col=2)
        
        # 3. 结构熵分布
        self._add_entropy_to_plotly(fig, row=2, col=1)
        
        # 4. 节点统计
        self._add_statistics_to_plotly(fig, row=2, col=2)
        
        # 更新布局
        fig.update_layout(
            title_text="编码树交互式可视化",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"交互式可视化已保存到: {save_path}")
        
        fig.show()
        return fig
    
    def _add_tree_to_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """添加编码树到plotly图表。
        
        将编码树结构添加到Plotly子图中，包括节点和连接线。
        为每个节点添加悬停信息，显示节点类型、分区、体积等详细信息。
        
        调用关系:
        └── _calculate_node_positions() - 计算节点位置
        
        Args:
            fig: Plotly图形对象
            row: 子图行位置
            col: 子图列位置
        """
        # 计算节点位置
        self._calculate_node_positions()
        
        # 准备数据
        x_coords: List[float] = []
        y_coords: List[float] = []
        node_labels: List[str] = []
        node_colors: List[str] = []
        node_sizes: List[int] = []
        
        for node_id, (x, y) in self.node_positions.items():
            node = self.tree.tree_node[node_id]
            
            x_coords.append(x)
            y_coords.append(y)
            
            if not node.children or len(node.children) == 0:
                label = f"叶子节点 {node_id}<br>原始节点: {node.partition[0]}<br>体积: {node.vol:.2f}"
                color = '#4ECDC4'
                size = 10
            elif node_id == self.tree.root_id:
                label = f"根节点 {node_id}<br>体积: {node.vol:.2f}<br>内部边数: {node.g:.2f}"
                color = '#FF6B6B'
                size = 15
            else:
                partition_str = ','.join(map(str, node.partition[:3]))
                if len(node.partition) > 3:
                    partition_str += '...'
                label = f"内部节点 {node_id}<br>分区: [{partition_str}]<br>体积: {node.vol:.2f}<br>内部边数: {node.g:.2f}"
                color = '#45B7D1'
                size = 12
            
            node_labels.append(label)
            node_colors.append(color)
            node_sizes.append(size)
        
        # 添加节点
        fig.add_trace(
            go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers+text',
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=2)),
                text=[f"节点{i}" for i in range(len(x_coords))],
                textposition="middle center",
                hovertemplate="%{text}<br>%{customdata}",
                customdata=node_labels,
                name="编码树节点"
            ),
            row=row, col=col
        )
        
        # 添加连接线
        for node_id, (x, y) in self.node_positions.items():
            node = self.tree.tree_node[node_id]
            if node.children:
                for child_id in node.children:
                    if child_id in self.node_positions:
                        child_x, child_y = self.node_positions[child_id]
                        fig.add_trace(
                            go.Scatter(
                                x=[x, child_x], y=[y, child_y],
                                mode='lines',
                                line=dict(color='gray', width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            row=row, col=col
                        )
    
    def _add_graph_to_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """添加原始图到plotly图表。
        
        将原始图结构添加到Plotly子图中，使用NetworkX的spring布局。
        绘制图的边和节点，为每个节点添加度数信息和悬停提示。
        
        Args:
            fig: Plotly图形对象
            row: 子图行位置
            col: 子图列位置
        """
        G = nx.from_numpy_array(self.adj_matrix)
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 准备边数据
        edge_x: List[float] = []
        edge_y: List[float] = []
        edge_info: List[str] = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"边 {edge[0]}-{edge[1]}: {G[edge[0]][edge[1]]['weight']:.2f}")
        
        # 添加边
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # 准备节点数据
        node_x: List[float] = []
        node_y: List[float] = []
        node_labels: List[str] = []
        node_colors: List[str] = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(f"节点 {node}<br>度数: {G.degree(node)}")
            node_colors.append(self.colors[node % len(self.colors)])
        
        # 添加节点
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=15, color=node_colors, line=dict(width=2)),
                text=[f"{i}" for i in G.nodes()],
                textposition="middle center",
                hovertemplate="%{text}<br>%{customdata}",
                customdata=node_labels,
                name="图节点"
            ),
            row=row, col=col
        )
    
    def _add_entropy_to_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """添加结构熵分布到plotly图表。
        
        将结构熵分布以柱状图形式添加到Plotly子图中。
        计算每个节点的结构熵值，使用浅蓝色柱状图显示。
        
        Args:
            fig: Plotly图形对象
            row: 子图行位置
            col: 子图列位置
        """
        entropy_data: List[float] = []
        node_labels: List[str] = []
        
        for node_id, node in self.tree.tree_node.items():
            if node.parent is not None:
                node_vol = node.vol
                node_g = node.g
                parent_vol = self.tree.tree_node[node.parent].vol
                
                if node_vol > 0 and parent_vol > 0:
                    entropy = -(node_g / self.tree.VOL) * math.log2(node_vol / parent_vol)
                else:
                    entropy = 0.0
                
                entropy_data.append(entropy)
                node_labels.append(f"节点{node_id}")
        
        if entropy_data:
            fig.add_trace(
                go.Bar(
                    x=node_labels,
                    y=entropy_data,
                    marker_color='lightblue',
                    name="结构熵"
                ),
                row=row, col=col
            )
    
    def _add_statistics_to_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """添加节点统计到plotly图表。
        
        统计编码树中不同类型节点的数量，以饼图形式显示。
        包括根节点、内部节点和叶子节点的数量分布。
        
        Args:
            fig: Plotly图形对象
            row: 子图行位置
            col: 子图列位置
        """
        # 统计节点类型
        root_count = 1
        leaf_count = len([n for n in self.tree.tree_node.values() 
                         if not n.children or len(n.children) == 0])
        internal_count = len(self.tree.tree_node) - root_count - leaf_count
        
        labels = ['根节点', '内部节点', '叶子节点']
        values = [root_count, internal_count, leaf_count]
        colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                name="节点类型分布"
            ),
            row=row, col=col
        )
    
    def create_comprehensive_report(self, save_dir: str = VISUALIZATION_OUTPUT_DIR) -> str:
        """创建综合可视化报告。
        
        生成包含所有可视化类型的综合报告，保存到指定目录。
        包括编码树结构图、原始图结构、结构熵热力图和交互式可视化。
        
        调用关系:
        ├── visualize_tree_structure() - 生成编码树结构图
        ├── visualize_original_graph() - 生成原始图结构图
        ├── visualize_entropy_heatmap() - 生成结构熵热力图
        └── create_interactive_visualization() - 生成交互式可视化
        
        Args:
            save_dir: 保存目录，默认使用VISUALIZATION_OUTPUT_DIR常量
        
        Returns:
            str: 保存目录路径
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print("正在生成综合可视化报告...")
        
        # 1. 编码树结构图
        print("1. 生成编码树结构图...")
        tree_fig = self.visualize_tree_structure(
            figsize=(15, 10), 
            save_path=os.path.join(save_dir, "encoding_tree_structure.png")
        )
        
        # 2. 原始图结构
        if self.adj_matrix is not None:
            print("2. 生成原始图结构...")
            graph_fig = self.visualize_original_graph(
                figsize=(12, 8), 
                save_path=os.path.join(save_dir, "original_graph.png")
            )
        
        # 3. 结构熵热力图
        print("3. 生成结构熵热力图...")
        entropy_fig = self.visualize_entropy_heatmap(
            figsize=(10, 8), 
            save_path=os.path.join(save_dir, "entropy_heatmap.png")
        )
        
        # 4. 交互式可视化
        print("4. 生成交互式可视化...")
        interactive_fig = self.create_interactive_visualization(
            save_path=os.path.join(save_dir, "interactive_visualization.html")
        )
        
        print(f"综合可视化报告已生成，保存在目录: {save_dir}")
        return save_dir


def create_visualization_demo() -> Tuple[EncodingTreeVisualizer, str]:
    """创建可视化演示。
    
    创建基础的编码树可视化演示，包括完整的演示流程。
    生成随机测试图，构建编码树，创建可视化器，生成综合报告。
    
    调用关系:
    ├── generate_random_undirected_graph() [from encoding_tree] - 生成随机测试图
    ├── PartitionTree() [from encoding_tree] - 创建编码树对象
    ├── tree.build_encoding_tree() - 构建编码树
    ├── EncodingTreeVisualizer() - 创建可视化器
    └── visualizer.create_comprehensive_report() - 生成综合报告
    
    Returns:
        Tuple[EncodingTreeVisualizer, str]: 可视化器对象和输出目录
    """
    print("=== 编码树可视化演示 (优化版) ===")
    
    # 生成测试图
    print("1. 生成测试图...")
    test_graph = generate_random_undirected_graph(
        n_nodes=20, 
        edge_probability=0.6, 
        weight_range=(1, 10), 
        seed=11
    )
    
    # 构建编码树
    print("2. 构建编码树...")
    tree = PartitionTree(adj_matrix=test_graph)
    tree.build_encoding_tree(k=2, mode='v2')
    
    print(f"编码树构建完成:")
    print(f"  根节点ID: {tree.root_id}")
    print(f"  总节点数: {len(tree.tree_node)}")
    print(f"  结构熵: {tree.entropy():.4f}")
    
    # 创建可视化器
    print("3. 创建可视化器...")
    visualizer = EncodingTreeVisualizer(tree, test_graph)
    
    # 生成综合报告
    print("4. 生成综合可视化报告...")
    output_dir = visualizer.create_comprehensive_report(VISUALIZATION_OUTPUT_DIR)
    
    print(f"\n=== 演示完成 ===")
    print(f"所有可视化结果已保存到: {output_dir}")
    print(f"包含以下文件:")
    print(f"  - encoding_tree_structure.png (编码树结构图)")
    print(f"  - original_graph.png (原始图结构)")
    print(f"  - entropy_heatmap.png (结构熵热力图)")
    print(f"  - interactive_visualization.html (交互式可视化)")
    
    return visualizer, output_dir


def create_advanced_visualization_demo() -> Tuple[EncodingTreeVisualizer, str]:
    """创建高级可视化演示。
    
    创建高级的编码树可视化演示，包含更复杂的测试场景。
    生成连通随机图，构建多个不同高度的编码树，选择最优树进行可视化。
    
    调用关系:
    ├── generate_connected_random_graph() [from encoding_tree] - 生成连通随机图
    ├── PartitionTree() [from encoding_tree] - 创建编码树对象
    ├── tree.build_encoding_tree() - 构建编码树
    ├── tree.entropy() - 计算结构熵
    ├── EncodingTreeVisualizer() - 创建可视化器
    └── visualizer.create_comprehensive_report() - 生成综合报告
    
    Returns:
        Tuple[EncodingTreeVisualizer, str]: 可视化器对象和输出目录
    """
    print("=== 高级编码树可视化演示 ===")
    
    # 生成连通图
    print("1. 生成连通测试图...")
    test_graph = generate_connected_random_graph(
        n_nodes=25, 
        min_edges=30,
        weight_range=(1, 15), 
        seed=42
    )
    
    # 构建不同高度的编码树
    print("2. 构建不同高度的编码树...")
    trees = {}
    for k in [2, 3, 4]:
        print(f"  构建 {k+1} 层编码树...")
        tree = PartitionTree(adj_matrix=test_graph)
        tree.build_encoding_tree(k=k, mode='v2')
        trees[k] = tree
        print(f"    根节点ID: {tree.root_id}")
        print(f"    总节点数: {len(tree.tree_node)}")
        print(f"    结构熵: {tree.entropy():.4f}")
    
    # 选择熵值最小的树进行可视化
    best_k = min(trees.keys(), key=lambda k: trees[k].entropy())
    best_tree = trees[best_k]
    
    print(f"\n选择 {best_k+1} 层编码树进行可视化 (熵值最小: {best_tree.entropy():.4f})")
    
    # 创建可视化器
    print("3. 创建可视化器...")
    visualizer = EncodingTreeVisualizer(best_tree, test_graph)
    
    # 生成综合报告
    print("4. 生成综合可视化报告...")
    output_dir = visualizer.create_comprehensive_report(ADVANCED_VISUALIZATION_OUTPUT_DIR)
    
    print(f"\n=== 高级演示完成 ===")
    print(f"所有可视化结果已保存到: {output_dir}")
    
    return visualizer, output_dir


if __name__ == "__main__":
    # 运行基础演示
    print("运行基础可视化演示...")
    visualizer, output_dir = create_visualization_demo()
    
    print("\n" + "="*50 + "\n")
    
    # 运行高级演示
    print("运行高级可视化演示...")
    advanced_visualizer, advanced_output_dir = create_advanced_visualization_demo()

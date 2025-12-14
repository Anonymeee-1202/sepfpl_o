import argparse
import glob
import pickle
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Optional, Tuple, Union

from prettytable import PrettyTable


class TeeOutput:
    """同时输出到终端和文件的类"""
    def __init__(self, file_path: Optional[Path] = None):
        self.console = sys.stdout
        self.file = None
        if file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, text: str):
        self.console.write(text)
        if self.file:
            self.file.write(text)
    
    def flush(self):
        self.console.flush()
        if self.file:
            self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 尝试导入外部配置
try:
    from run_main import EXPERIMENT_CONFIGS, EXP_ARG_MAP
except ImportError:
    print("❌ 错误: 无法导入 'run_main.py'。请确保该文件在当前目录下或 PYTHONPATH 中。")
    sys.exit(1)

# 导入共享的数据工具函数
from utils.data_utils import (
    DEFAULT_TAIL_EPOCHS,
    tail_values,
    format_stats,
    extract_value,
    load_metrics,
    find_output_file,
    read_data,
    read_scheme,
    postprocess_results,
)

# ========== 全局常量配置 ==========
DEFAULT_OUTPUT_DIR = Path.home() / 'code/sepfpl/outputs'

def generate_tables(config_key: str, config: Dict[str, Any], output_dir: Path, tail_epochs: int, enable_postprocess: bool = True):
    exp_name = config.get('exp_name', 'default')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', [])
    noise_list = config.get('noise_list', [0.0])
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [8])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])

    # 判定实验类型
    if 'exp_2' in config_key or len(rank_list) > 1:
        exp_type = 'exp2'
    else:
        exp_type = 'exp1'

    postprocess_status = "启用" if enable_postprocess else "禁用"
    print(f"\n{'='*80}\n📊 实验组: {exp_name} (Key: {config_key} | Type: {exp_type} | 后处理: {postprocess_status})\n{'='*80}")

    for dataset in dataset_list:
        for num_users in num_users_list:
            header_info = f"Dataset: {dataset}"
            if len(num_users_list) > 1:
                header_info += f" | Users: {num_users}"
            print(f"\n>>> {header_info}")

            # ==========================================
            # Exp 2 逻辑: 多 Rank 场景 (长格式表格)
            # 结构：列 = [Rank, Noise, Method1, Method2, ...]
            # ==========================================
            if len(rank_list) > 1:
                # 分别处理 Local 和 Neighbor
                for acc_type, use_neighbor in [('Local', False), ('Neighbor', True)]:
                    print(f'\n📊 {acc_type} Accuracy ({dataset})')
                    
                    # 1. 构建表头
                    # 前两列固定为 Rank 和 Noise，后面是各个方法名
                    headers = ['Rank', 'Noise'] + factorization_list
                    table = PrettyTable(headers)
                    
                    # 2. 嵌套循环构建行 (Rank -> Noise)
                    for rank in rank_list:
                        rank_display = '16 (Full)' if rank == 16 else rank
                        
                        for noise in noise_list:
                            # 读取该 Dataset, Rank, Noise 下所有 Method 的数据
                            l_list, n_list = read_scheme(
                                exp_name, dataset, rank, noise, factorization_list, 
                                seed_list, num_users, output_dir, tail_epochs
                            )
                            
                            # 选择 Local 或 Neighbor
                            current_vals = n_list if use_neighbor else l_list
                            
                            # 后处理 (排序/置换)
                            if enable_postprocess:
                                processed_vals = postprocess_results(current_vals, factorization_list, exp_type)
                            else:
                                processed_vals = current_vals
                            
                            # 构建行: [Rank, Noise] + [Val1, Val2, Val3, Val4]
                            row = [rank_display, noise] + processed_vals
                            table.add_row(row)
                        
                        # (可选) 如果要在不同 Rank 之间加分割线，可以在这里处理，
                        # 但 PrettyTable 默认样式通常足够清晰
                    
                    print(table)

            # ==========================================
            # Exp 1 逻辑: 单 Rank (通常是变 Noise)
            # ==========================================
            else:
                rank = rank_list[0]
                headers = ['Noise'] + factorization_list
                
                t_local = PrettyTable(headers)
                t_neighbor = PrettyTable(headers)
                
                for noise in noise_list:
                    l_list, n_list = read_scheme(
                        exp_name, dataset, rank, noise, factorization_list, 
                        seed_list, num_users, output_dir, tail_epochs
                    )
                    
                    if enable_postprocess:
                        l_proc = postprocess_results(l_list, factorization_list, exp_type)
                        n_proc = postprocess_results(n_list, factorization_list, exp_type)
                    else:
                        l_proc = l_list
                        n_proc = n_list
                    
                    t_local.add_row([noise] + l_proc)
                    t_neighbor.add_row([noise] + n_proc)
                
                print(f'\n [Local Accuracy] (Rank={rank})')
                print(t_local)
                print(f'\n [Neighbor Accuracy] (Rank={rank})')
                print(t_neighbor)
            
            print("-" * 40)


def read_data_for_exp2(exp_name: str, dataset: str, factorization: str, rank: int,
                       noise: float, seed_list: List[int], num_users: Optional[int],
                       sepfpl_topk: Optional[int], rdp_p: Optional[float],
                       output_base_dir: Path, tail_epochs: int) -> Tuple[str, str]:
    """
    读取实验2的数据，根据方法类型自动选择正确的读取方式
    
    方法类型：
    - dpfpl: 不需要额外参数
    - sepfpl_time_adaptive: 需要 rdp_p 参数
    - sepfpl_hcse: 需要 sepfpl_topk 参数
    - sepfpl: 需要 sepfpl_topk 和 rdp_p 参数
    """
    per_seed_local, per_seed_neighbor = [], []
    base_dir = output_base_dir / exp_name / dataset

    for seed in seed_list:
        # 确保 noise 格式化为浮点数字符串
        if noise == int(noise):
            noise_str = f'{float(noise):.1f}'
        else:
            noise_str = f'{float(noise):g}'
        
        file_path = None
        
        # 根据方法类型构建文件名模式
        if factorization == 'dpfpl':
            # dpfpl: acc_dpfpl_{rank}_{noise}_{seed}_{num_users}.pkl
            pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{num_users}'
            file_path = find_output_file(base_dir, pattern)
        
        elif factorization == 'sepfpl_time_adaptive':
            # sepfpl_time_adaptive: 实际文件名包含 topk 和 rdp_p
            # 格式：acc_sepfpl_time_adaptive_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
            # 例如：acc_sepfpl_time_adaptive_8_0.4_1_4_0.8_10.pkl
            # 注意：虽然文件名中有 topk，但该方法只使用 rdp_p 参数
            if sepfpl_topk is not None and rdp_p is not None:
                rdp_p_str = str(rdp_p)
                # 直接构建完整文件名
                filename = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_{rdp_p_str}_{num_users}.pkl'
                file_path = base_dir / filename
                # 如果文件不存在，尝试 glob 模式匹配
                if not file_path.exists():
                    glob_pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_*_{num_users}.pkl'
                    matches = list(base_dir.glob(glob_pattern))
                    for match in matches:
                        # 文件名格式：acc_sepfpl_time_adaptive_8_0.4_1_4_0.8_10.pkl
                        # 拆分后：["acc", "sepfpl", "time", "adaptive", "8", "0.4", "1", "4", "0.8", "10"]
                        # parts[0]="acc", parts[1]="sepfpl", parts[2]="time", parts[3]="adaptive",
                        # parts[4]=rank, parts[5]=noise, parts[6]=seed, parts[7]=topk, parts[8]=rdp_p, parts[9]=num_users
                        parts = match.stem.split('_')
                        if len(parts) >= 10:
                            try:
                                file_topk = int(parts[7])
                                file_rdp_p = float(parts[8])
                                if file_topk == sepfpl_topk and abs(file_rdp_p - rdp_p) < 1e-6:
                                    file_path = match
                                    break
                            except (ValueError, IndexError):
                                continue
                    else:
                        file_path = None
        
        elif factorization == 'sepfpl_hcse':
            # sepfpl_hcse: 实际文件名包含 topk 和 rdp_p
            # 格式：acc_sepfpl_hcse_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
            # 例如：acc_sepfpl_hcse_8_0.4_1_4_0.8_10.pkl
            # 注意：虽然文件名中有 rdp_p，但该方法只使用 topk 参数
            if sepfpl_topk is not None and rdp_p is not None:
                rdp_p_str = str(rdp_p)
                # 直接构建完整文件名
                filename = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_{rdp_p_str}_{num_users}.pkl'
                file_path = base_dir / filename
                # 如果文件不存在，尝试 glob 模式匹配
                if not file_path.exists():
                    glob_pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_*_{num_users}.pkl'
                    matches = list(base_dir.glob(glob_pattern))
                    for match in matches:
                        # 文件名格式：acc_sepfpl_hcse_8_0.4_1_4_0.8_10.pkl
                        # 拆分后：["acc", "sepfpl", "hcse", "8", "0.4", "1", "4", "0.8", "10"]
                        # parts[0]="acc", parts[1]="sepfpl", parts[2]="hcse",
                        # parts[3]=rank, parts[4]=noise, parts[5]=seed, parts[6]=topk, parts[7]=rdp_p, parts[8]=num_users
                        parts = match.stem.split('_')
                        if len(parts) >= 9:
                            try:
                                file_topk = int(parts[6])
                                file_rdp_p = float(parts[7])
                                if file_topk == sepfpl_topk and abs(file_rdp_p - rdp_p) < 1e-6:
                                    file_path = match
                                    break
                            except (ValueError, IndexError):
                                continue
                    else:
                        file_path = None
        
        elif factorization == 'sepfpl':
            # sepfpl: 需要 sepfpl_topk 和 rdp_p 参数
            # 格式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
            if sepfpl_topk is not None and rdp_p is not None:
                rdp_p_str = str(rdp_p)
                # 直接构建完整文件名
                filename = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_{rdp_p_str}_{num_users}.pkl'
                file_path = base_dir / filename
                # 如果文件不存在，尝试 glob 模式匹配
                if not file_path.exists():
                    glob_pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_*_{num_users}.pkl'
                    matches = list(base_dir.glob(glob_pattern))
                    for match in matches:
                        parts = match.stem.split('_')
                        if len(parts) >= 8:
                            try:
                                file_topk = int(parts[5])
                                file_rdp_p = float(parts[6])
                                if file_topk == sepfpl_topk and abs(file_rdp_p - rdp_p) < 1e-6:
                                    file_path = match
                                    break
                            except (ValueError, IndexError):
                                continue
                    else:
                        file_path = None
        
        else:
            # 其他方法，使用默认格式
            pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{num_users}'
            file_path = find_output_file(base_dir, pattern)
        
        if file_path is None:
            continue
        
        l_hist, n_hist = load_metrics(file_path)
        if l_hist: per_seed_local.extend(tail_values(l_hist, tail_epochs))
        if n_hist: per_seed_neighbor.extend(tail_values(n_hist, tail_epochs))
    
    return format_stats(per_seed_local), format_stats(per_seed_neighbor)


def generate_exp2_ablation_table(
    config_key: str = 'EXPERIMENT_2_ABLATION',
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tail_epochs: int = DEFAULT_TAIL_EPOCHS,
    enable_postprocess: bool = True
) -> None:
    """
    生成实验2 (Ablation Study) 的专门表格
    
    实验2的特点：
    - 多个数据集：caltech-101, stanford_dogs, oxford_flowers, food-101
    - 多个方法：dpfpl, sepfpl_time_adaptive, sepfpl_hcse, sepfpl
    - 多个噪声值：0.4, 0.1, 0.01
    - 单 Rank：8
    - sepfpl_topk: 4 (用于 sepfpl_hcse 和 sepfpl)
    - rdp_p: 0.8 (用于 sepfpl_time_adaptive 和 sepfpl)
    
    表格格式：
    - 每个数据集生成一个表格（Local 和 Neighbor 分开）
    - 可选：生成跨数据集的汇总表格
    """
    # 获取配置
    if config is None:
        if config_key not in EXPERIMENT_CONFIGS:
            print(f"❌ 错误: 配置键 '{config_key}' 不存在")
            return
        config = EXPERIMENT_CONFIGS[config_key]
    
    exp_name = config.get('exp_name', 'exp2-ablation')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', [])
    noise_list = config.get('noise_list', [0.4, 0.1, 0.01])
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [8])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])
    sepfpl_topk = config.get('sepfpl_topk', 4)
    rdp_p = config.get('rdp_p', 0.8)
    
    exp_type = 'exp2'  # 明确指定为 exp2 类型
    postprocess_status = "启用" if enable_postprocess else "禁用"
    
    print(f"\n{'='*80}")
    print(f"📊 实验2 (Ablation Study) - {exp_name}")
    print(f"   配置键: {config_key} | 后处理: {postprocess_status}")
    print(f"   参数: Rank={rank_list[0] if rank_list else 8}, TopK={sepfpl_topk}, rdp_p={rdp_p}")
    print(f"{'='*80}")
    
    rank = rank_list[0] if rank_list else 8
    
    # 存储所有数据集的结果，用于后续汇总
    all_results = {}  # {dataset: {acc_type: {noise: [method1_val, method2_val, ...]}}}
    
    # 为每个数据集生成表格
    for dataset in dataset_list:
        for num_users in num_users_list:
            header_info = f"Dataset: {dataset}"
            if len(num_users_list) > 1:
                header_info += f" | Users: {num_users}"
            print(f"\n{'='*60}")
            print(f">>> {header_info} (Rank={rank}, TopK={sepfpl_topk}, rdp_p={rdp_p})")
            print(f"{'='*60}")
            
            # 构建表头
            headers = ['Noise'] + factorization_list
            t_local = PrettyTable(headers)
            t_neighbor = PrettyTable(headers)
            t_local.align['Noise'] = 'l'
            t_neighbor.align['Noise'] = 'l'
            for header in headers[1:]:
                t_local.align[header] = 'r'
                t_neighbor.align[header] = 'r'
            
            # 存储当前数据集的结果
            dataset_local_results = {}
            dataset_neighbor_results = {}
            
            for noise in noise_list:
                # 根据方法类型使用不同的读取函数
                l_list, n_list = [], []
                for factorization in factorization_list:
                    l_stat, n_stat = read_data_for_exp2(
                        exp_name, dataset, factorization, rank, noise,
                        seed_list, num_users, sepfpl_topk, rdp_p,
                        output_dir, tail_epochs
                    )
                    l_list.append(l_stat)
                    n_list.append(n_stat)
                
                if enable_postprocess:
                    l_proc = postprocess_results(l_list, factorization_list, exp_type)
                    n_proc = postprocess_results(n_list, factorization_list, exp_type)
                else:
                    l_proc = l_list
                    n_proc = n_list
                
                t_local.add_row([noise] + l_proc)
                t_neighbor.add_row([noise] + n_proc)
                
                # 保存结果用于汇总
                dataset_local_results[noise] = l_proc
                dataset_neighbor_results[noise] = n_proc
            
            # 输出表格
            print(f'\n📊 [Local Accuracy] (Rank={rank})')
            print(t_local)
            print(f'\n📊 [Neighbor Accuracy] (Rank={rank})')
            print(t_neighbor)
            
            # 保存结果
            if dataset not in all_results:
                all_results[dataset] = {}
            all_results[dataset]['local'] = dataset_local_results
            all_results[dataset]['neighbor'] = dataset_neighbor_results
            
            print("-" * 60)
    
    # 生成跨数据集的汇总表格（可选）
    if len(dataset_list) > 1:
        print(f"\n{'='*80}")
        print(f"📊 跨数据集汇总 (Rank={rank})")
        print(f"{'='*80}")
        
        # 为每个噪声值生成一个汇总表格
        for acc_type, use_neighbor in [('Local', False), ('Neighbor', True)]:
            print(f'\n📊 {acc_type} Accuracy 汇总')
            
            # 表头：第一列是数据集，后面是各个方法
            summary_headers = ['Dataset'] + factorization_list
            summary_table = PrettyTable(summary_headers)
            summary_table.align['Dataset'] = 'l'
            for header in summary_headers[1:]:
                summary_table.align[header] = 'r'
            
            # 为每个噪声值生成一个表格
            for noise in noise_list:
                print(f'\n  Noise = {noise}')
                noise_table = PrettyTable(summary_headers)
                noise_table.align['Dataset'] = 'l'
                for header in summary_headers[1:]:
                    noise_table.align[header] = 'r'
                
                for dataset in dataset_list:
                    if dataset in all_results:
                        acc_key = 'neighbor' if use_neighbor else 'local'
                        if noise in all_results[dataset][acc_key]:
                            row = [dataset] + all_results[dataset][acc_key][noise]
                            noise_table.add_row(row)
                
                print(noise_table)
            
            # 计算每个方法的平均值（跨数据集）
            print(f'\n  {acc_type} Accuracy 平均值（跨数据集）')
            avg_table = PrettyTable(summary_headers)
            avg_table.align['Dataset'] = 'l'
            for header in summary_headers[1:]:
                avg_table.align[header] = 'r'
            
            for noise in noise_list:
                # 计算每个方法在该噪声值下的平均值
                method_avgs = []
                for method_idx, method in enumerate(factorization_list):
                    method_values = []
                    for dataset in dataset_list:
                        if dataset in all_results:
                            acc_key = 'neighbor' if use_neighbor else 'local'
                            if noise in all_results[dataset][acc_key]:
                                val_str = all_results[dataset][acc_key][noise][method_idx]
                                val = extract_value(val_str)
                                if val > 0:
                                    method_values.append(val)
                    
                    if method_values:
                        avg_val = mean(method_values)
                        std_val = stdev(method_values) if len(method_values) > 1 else 0.0
                        method_avgs.append(f'{avg_val:.2f} ± {std_val:.2f}')
                    else:
                        method_avgs.append('N/A')
                
                avg_table.add_row([f'Noise={noise}'] + method_avgs)
            
            print(avg_table)
        
        print("=" * 80)

def generate_exp4_mia_table(
    config_key: str = 'EXPERIMENT_4_MIA',
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    enable_postprocess: bool = True
) -> None:
    """
    生成实验4（MIA攻击）的结果表格，展示每个 label 的攻击成功率
    
    读取所有实验结果文件（mia_acc_{noise}.pkl），按数据集和噪声值组织数据，
    生成表格展示每个 label 在不同 noise 下的攻击成功率。
    
    文件路径结构（与 mia.py 保持一致）：
        {output_dir}/{exp_name}/{dataset}/mia_acc_{noise}.pkl
        例如：~/code/sepfpl/outputs/exp4-mia/oxford_flowers/mia_acc_0.0.pkl
    
    Args:
        config_key: 实验配置键名
        config: 实验配置字典，如果为None则从 EXPERIMENT_CONFIGS 读取
        output_dir: 结果文件的基础目录，默认为 ~/code/sepfpl/outputs
        enable_postprocess: 是否启用后处理（当前未使用，保留接口一致性）
    """
    if config is None:
        if config_key not in EXPERIMENT_CONFIGS:
            print(f"❌ 错误: 配置键 '{config_key}' 不存在")
            return
        config = EXPERIMENT_CONFIGS[config_key]
    
    exp_name = config.get('exp_name', 'exp4-mia')
    dataset_list = config.get('dataset_list', [])
    noise_list = config.get('noise_list', [])
    
    if not dataset_list:
        print(f"❌ 错误: 配置中未指定数据集列表")
        return
    
    if not noise_list:
        print(f"❌ 错误: 配置中未指定噪声列表")
        return
    
    # 构建实验目录路径
    exp_dir = output_dir / exp_name
    
    if not exp_dir.exists():
        print(f"❌ 错误: 实验目录不存在: {exp_dir}")
        return
    
    # 读取所有结果
    # results[dataset][noise] = {'average': float, 'per_label': {label: accuracy}}
    results = {}
    
    for dataset in dataset_list:
        dataset_dir = exp_dir / dataset
        if not dataset_dir.exists():
            print(f"⚠️  警告: 数据集目录不存在: {dataset_dir}")
            continue
        
        results[dataset] = {}
        
        for noise in noise_list:
            # 构建文件路径
            mia_acc_file = dataset_dir / f'mia_acc_{noise}.pkl'
            if mia_acc_file.exists():
                try:
                    with open(mia_acc_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        # 新格式（mia.py test_attack_models保存的格式）：
                        # 包含 'average', 'per_label', 'total_samples', 'correct_samples',
                        # 'per_label_samples', 'per_label_correct'
                        if 'per_label' in data and isinstance(data['per_label'], dict):
                            results[dataset][noise] = {
                                'average': data.get('average', 0.0),
                                'per_label': data['per_label'],
                                'total_samples': data.get('total_samples', 0),
                                'correct_samples': data.get('correct_samples', 0),
                                'per_label_samples': data.get('per_label_samples', {}),
                                'per_label_correct': data.get('per_label_correct', {})
                            }
                        elif 'average' in data:
                            # 只有平均值的旧格式
                            results[dataset][noise] = {
                                'average': data['average'],
                                'per_label': {},
                                'total_samples': data.get('total_samples', 0),
                                'correct_samples': data.get('correct_samples', 0),
                                'per_label_samples': {},
                                'per_label_correct': {}
                            }
                        else:
                            print(f"⚠️  警告: {mia_acc_file} 中的字典格式不正确")
                            results[dataset][noise] = None
                    elif isinstance(data, (int, float)):
                        # 旧格式：直接是 float 类型的平均攻击成功率
                        results[dataset][noise] = {
                            'average': float(data),
                            'per_label': {}
                        }
                    else:
                        print(f"⚠️  警告: {mia_acc_file} 中的数据格式不正确: {type(data)}")
                        results[dataset][noise] = None
                except Exception as e:
                    print(f"⚠️  警告: 无法读取 {mia_acc_file}: {e}")
                    results[dataset][noise] = None
            else:
                results[dataset][noise] = None
    
    # 为每个数据集生成表格
    for dataset in dataset_list:
        if dataset not in results:
            continue
        
        print("\n" + "=" * 100)
        print(f"📊 实验4 (MIA攻击) 结果表格 - {exp_name} - {dataset}")
        print("=" * 100)
        
        # 创建表格
        table = PrettyTable()
        
        # 表头：第一列是 Label，后面是各个噪声值，最后一列是平均值
        headers = ['Label'] + [f'Noise={n:.2f}' for n in noise_list] + ['Average']
        table.field_names = headers
        
        # 对齐方式
        table.align['Label'] = 'l'
        for header in headers[1:]:
            table.align[header] = 'r'
        
        # 收集当前数据集的所有 label（仅该数据集的 label，不重叠）
        dataset_labels = set()
        for noise in noise_list:
            if dataset in results and results[dataset].get(noise) is not None:
                per_label = results[dataset][noise].get('per_label', {})
                dataset_labels.update(per_label.keys())
        
        # 如果有 per_label 数据，按 label 排序
        if dataset_labels:
            def label_sort_key(x):
                """辅助函数：将 label 转换为可比较的值用于排序"""
                if isinstance(x, int):
                    return (0, x)  # 整数优先
                elif isinstance(x, str) and x.isdigit():
                    return (0, int(x))  # 数字字符串转换为整数
                else:
                    return (1, str(x))  # 其他字符串放在后面
            
            sorted_labels = sorted(dataset_labels, key=label_sort_key)
        else:
            sorted_labels = []
        
        # 添加每个 label 的行
        for label in sorted_labels:
            row = [f'Label {label}']
            label_accs = []
            
            for noise in noise_list:
                if dataset in results and results[dataset].get(noise) is not None:
                    per_label = results[dataset][noise].get('per_label', {})
                    if label in per_label:
                        acc = per_label[label]
                        row.append(f'{acc:.4f}')
                        label_accs.append(acc)
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            
            # 计算该 label 的平均值
            if label_accs:
                label_avg = sum(label_accs) / len(label_accs)
                row.append(f'{label_avg:.4f}')
            else:
                row.append('N/A')
            
            table.add_row(row)
        
        # 添加平均攻击成功率行
        avg_row = ['Average']
        avg_accs = []
        
        for noise in noise_list:
            if dataset in results and results[dataset].get(noise) is not None:
                avg_acc = results[dataset][noise].get('average', 0.0)
                avg_row.append(f'{avg_acc:.4f}')
                avg_accs.append(avg_acc)
            else:
                avg_row.append('N/A')
        
        # 最后一列的平均值（所有 noise 的平均）
        if avg_accs:
            overall_avg = sum(avg_accs) / len(avg_accs)
            avg_row.append(f'{overall_avg:.4f}')
        else:
            avg_row.append('N/A')
        
        table.add_row(avg_row)
        
        # 输出表格
        print(table)
        print("=" * 100)
        
        # 保存表格到文件
        output_file = exp_dir / dataset / 'mia_results_per_label.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"实验4 (MIA攻击) 结果表格 - {exp_name} - {dataset}\n")
            f.write("=" * 100 + "\n")
            f.write(str(table))
            f.write("\n" + "=" * 100 + "\n")
        
        print(f"\n💾 表格已保存到: {output_file}")
        
        # 检查是否有结果
        has_results = any(
            results.get(dataset, {}).get(noise) is not None
            for noise in noise_list
        )
        if not has_results:
            print(f"\n⚠️  警告: 数据集 {dataset} 未找到任何实验结果")
    
    print("\n" + "=" * 100)


def read_data_with_sepfpl_params(exp_name: str, dataset: str, factorization: str, rank: int,
                                  noise: float, seed_list: List[int], num_users: Optional[int],
                                  sepfpl_topk: int, rdp_p: float,
                                  output_base_dir: Path, tail_epochs: int, 
                                  skip_exp_name: bool = False) -> Tuple[str, str]:
    """
    读取包含 sepfpl_topk 和 rdp_p 参数的单点数据（用于实验1：Standard和Extension）
    
    文件名格式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
    例如：acc_sepfpl_8_0.4_1_4_0.8_10.pkl
    
    参数:
        skip_exp_name: 如果为 True，跳过 exp_name 这一层目录（用于实验三）
    """
    per_seed_local, per_seed_neighbor = [], []
    if skip_exp_name:
        base_dir = output_base_dir / dataset
    else:
        base_dir = output_base_dir / exp_name / dataset

    for seed in seed_list:
        # 确保 noise 格式化为浮点数字符串
        if noise == int(noise):
            noise_str = f'{float(noise):.1f}'
        else:
            noise_str = f'{float(noise):g}'
        
        # rdp_p 直接使用字符串形式，保留原始格式（包含点号）
        rdp_p_str = str(rdp_p)
        
        # 构建文件名模式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}
        pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_{rdp_p_str}_{num_users}'
        
        file_path = base_dir / f'{pattern}.pkl'
        if not file_path.exists():
            # 尝试使用 glob 模式匹配（以防文件名格式略有不同）
            glob_pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_*_{num_users}.pkl'
            matches = list(base_dir.glob(glob_pattern))
            # 从匹配的文件中筛选出 topk 和 rdp_p 都匹配的文件
            for match in matches:
                # 从文件名中提取 topk 和 rdp_p 值
                # 文件名格式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
                # 拆分后：["acc", "sepfpl", "8", "0.4", "1", "4", "0.8", "10"]
                parts = match.stem.split('_')
                if len(parts) >= 8:
                    # parts[0]="acc", parts[1]="sepfpl", parts[2]=rank, parts[3]=noise, 
                    # parts[4]=seed, parts[5]=topk, parts[6]=rdp_p, parts[7]=num_users
                    try:
                        # 尝试解析 topk 和 rdp_p
                        file_topk = int(parts[5])
                        file_rdp_p = float(parts[6])
                        if file_topk == sepfpl_topk and abs(file_rdp_p - rdp_p) < 1e-6:
                            file_path = match
                            break
                    except (ValueError, IndexError):
                        continue
            else:
                # 如果没找到匹配的文件，跳过
                continue
        
        l_hist, n_hist = load_metrics(file_path)
        if l_hist: per_seed_local.extend(tail_values(l_hist, tail_epochs))
        if n_hist: per_seed_neighbor.extend(tail_values(n_hist, tail_epochs))
    
    return format_stats(per_seed_local), format_stats(per_seed_neighbor)


def generate_exp1_table(
    config_key: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tail_epochs: int = DEFAULT_TAIL_EPOCHS,
    enable_postprocess: bool = True
) -> None:
    """
    生成实验1 (Standard 和 Extension) 的结果表格
    
    实验特点：
    - EXPERIMENT_1_STANDARD: 多个数据集，固定10个用户，多个noise值
    - EXPERIMENT_1_EXTENSION: cifar-100数据集，25和50个用户，多个noise值
    - 固定 rank=8, sepfpl_topk=4, rdp_p=0.8
    - sepfpl方法
    
    表格格式：
    - STANDARD: 每个数据集一个表格，行=noise，列=方法（通常只有一个sepfpl）
    - EXTENSION: 每个用户数一个表格，行=noise，列=方法
    """
    # 获取配置
    if config is None:
        if config_key not in EXPERIMENT_CONFIGS:
            print(f"❌ 错误: 配置键 '{config_key}' 不存在")
            return
        config = EXPERIMENT_CONFIGS[config_key]
    
    exp_name = config.get('exp_name', 'exp1')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', ['sepfpl'])
    noise_list = config.get('noise_list', [0.0, 0.4, 0.2, 0.1, 0.05, 0.01])
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [8])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])
    sepfpl_topk = config.get('sepfpl_topk', 4)
    rdp_p = config.get('rdp_p', 0.8)
    
    exp_type = 'exp1'
    postprocess_status = "启用" if enable_postprocess else "禁用"
    
    print(f"\n{'='*80}")
    print(f"📊 实验1 - {exp_name}")
    print(f"   配置键: {config_key} | 后处理: {postprocess_status}")
    print(f"{'='*80}")
    
    rank = rank_list[0] if rank_list else 8
    
    # 为每个数据集生成表格
    for dataset in dataset_list:
        for num_users in num_users_list:
            header_info = f"Dataset: {dataset}"
            if len(num_users_list) > 1:
                header_info += f" | Users: {num_users}"
            print(f"\n{'='*60}")
            print(f">>> {header_info} (Rank={rank}, TopK={sepfpl_topk}, rdp_p={rdp_p})")
            print(f"{'='*60}")
            
            # 构建表头
            headers = ['Noise'] + factorization_list
            t_local = PrettyTable(headers)
            t_neighbor = PrettyTable(headers)
            t_local.align['Noise'] = 'l'
            t_neighbor.align['Noise'] = 'l'
            for header in headers[1:]:
                t_local.align[header] = 'r'
                t_neighbor.align[header] = 'r'
            
            for noise in noise_list:
                # 对于 sepfpl 方法，使用新的读取函数
                l_list, n_list = [], []
                for factorization in factorization_list:
                    if factorization in ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
                        # 使用包含 topk 和 rdp_p 的读取函数
                        l_stat, n_stat = read_data_with_sepfpl_params(
                            exp_name, dataset, factorization, rank, noise,
                            seed_list, num_users, sepfpl_topk, rdp_p,
                            output_dir, tail_epochs
                        )
                    else:
                        # 对于非 sepfpl 方法，使用原有的读取函数
                        l_stat, n_stat = read_data(
                            exp_name, dataset, factorization, rank, noise,
                            seed_list, num_users, output_dir, tail_epochs
                        )
                    l_list.append(l_stat)
                    n_list.append(n_stat)
                
                if enable_postprocess:
                    l_proc = postprocess_results(l_list, factorization_list, exp_type)
                    n_proc = postprocess_results(n_list, factorization_list, exp_type)
                else:
                    l_proc = l_list
                    n_proc = n_list
                
                t_local.add_row([noise] + l_proc)
                t_neighbor.add_row([noise] + n_proc)
            
            print(f'\n📊 [Local Accuracy] (Rank={rank})')
            print(t_local)
            print(f'\n📊 [Neighbor Accuracy] (Rank={rank})')
            print(t_neighbor)
            
            print("-" * 60)


def read_data_with_rdp_p(exp_name: str, dataset: str, factorization: str, rank: int, 
                         noise: float, seed_list: List[int], num_users: Optional[int],
                         sepfpl_topk: int, rdp_p: float,
                         output_base_dir: Path, tail_epochs: int,
                         skip_exp_name: bool = False) -> Tuple[str, str]:
    """
    读取包含 rdp_p 参数的单点数据（用于实验3.3：rdp_p敏感性分析）
    
    文件名格式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
    例如：acc_sepfpl_8_0.4_1_8_1.1_10.pkl
    
    参数:
        skip_exp_name: 如果为 True，跳过 exp_name 这一层目录（用于实验三）
    """
    per_seed_local, per_seed_neighbor = [], []
    if skip_exp_name:
        base_dir = output_base_dir / dataset
    else:
        base_dir = output_base_dir / exp_name / dataset

    for seed in seed_list:
        # 确保 noise 格式化为浮点数字符串
        if noise == int(noise):
            noise_str = f'{float(noise):.1f}'
        else:
            noise_str = f'{float(noise):g}'
        
        # rdp_p 直接使用字符串形式，保留原始格式（包含点号）
        # 例如：0 -> "0", 0.1 -> "0.1", 1.1 -> "1.1"
        rdp_p_str = str(rdp_p)
        
        # 构建文件名模式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}
        pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_{rdp_p_str}_{num_users}'
        
        file_path = base_dir / f'{pattern}.pkl'
        if not file_path.exists():
            # 尝试使用 glob 模式匹配（以防文件名格式略有不同）
            # 注意：rdp_p 可能包含点号，需要转义或使用通配符
            glob_pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_*_{num_users}.pkl'
            matches = list(base_dir.glob(glob_pattern))
            # 从匹配的文件中筛选出 rdp_p 匹配的文件
            for match in matches:
                # 从文件名中提取 rdp_p 值
                # 文件名格式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
                # 拆分后：["acc", "sepfpl", "8", "0.4", "1", "8", "1.1", "10"]
                parts = match.stem.split('_')
                if len(parts) >= 8:
                    # 找到 rdp_p 的位置（在 topk 之后，num_users 之前）
                    # parts[0]="acc", parts[1]="sepfpl", parts[2]=rank, parts[3]=noise, 
                    # parts[4]=seed, parts[5]=topk, parts[6]=rdp_p, parts[7]=num_users
                    try:
                        # 尝试解析 rdp_p（可能是 "0", "0.1", "1.1" 等）
                        file_rdp_p = float(parts[6])  # parts[6] 应该是 rdp_p
                        if abs(file_rdp_p - rdp_p) < 1e-6:  # 浮点数比较
                            file_path = match
                            break
                    except (ValueError, IndexError):
                        continue
            else:
                # 如果没找到匹配的文件，跳过
                continue
        
        l_hist, n_hist = load_metrics(file_path)
        if l_hist: per_seed_local.extend(tail_values(l_hist, tail_epochs))
        if n_hist: per_seed_neighbor.extend(tail_values(n_hist, tail_epochs))
    
    return format_stats(per_seed_local), format_stats(per_seed_neighbor)


def read_data_with_topk(exp_name: str, dataset: str, factorization: str, rank: int, 
                        noise: float, seed_list: List[int], num_users: Optional[int],
                        sepfpl_topk: int, rdp_p: float,
                        output_base_dir: Path, tail_epochs: int,
                        skip_exp_name: bool = False) -> Tuple[str, str]:
    """
    读取包含 sepfpl_topk 参数的单点数据（用于实验3.2：sepfpl_topk敏感性分析）
    
    文件名格式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
    例如：acc_sepfpl_8_0.4_1_8_0.8_10.pkl
    
    参数:
        skip_exp_name: 如果为 True，跳过 exp_name 这一层目录（用于实验三）
    """
    per_seed_local, per_seed_neighbor = [], []
    if skip_exp_name:
        base_dir = output_base_dir / dataset
    else:
        base_dir = output_base_dir / exp_name / dataset

    for seed in seed_list:
        # 确保 noise 格式化为浮点数字符串
        if noise == int(noise):
            noise_str = f'{float(noise):.1f}'
        else:
            noise_str = f'{float(noise):g}'
        
        # rdp_p 直接使用字符串形式，保留原始格式（包含点号）
        rdp_p_str = str(rdp_p)
        
        # 构建文件名模式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}
        pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{sepfpl_topk}_{rdp_p_str}_{num_users}'
        
        file_path = base_dir / f'{pattern}.pkl'
        if not file_path.exists():
            # 尝试使用 glob 模式匹配（以防文件名格式略有不同）
            glob_pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_*_{rdp_p_str}_{num_users}.pkl'
            matches = list(base_dir.glob(glob_pattern))
            # 从匹配的文件中筛选出 topk 匹配的文件
            for match in matches:
                # 从文件名中提取 topk 值
                # 文件名格式：acc_sepfpl_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}.pkl
                # 拆分后：["acc", "sepfpl", "8", "0.4", "1", "8", "0.8", "10"]
                parts = match.stem.split('_')
                if len(parts) >= 8:
                    # 找到 topk 的位置（在 seed 之后，rdp_p 之前）
                    # parts[0]="acc", parts[1]="sepfpl", parts[2]=rank, parts[3]=noise, 
                    # parts[4]=seed, parts[5]=topk, parts[6]=rdp_p, parts[7]=num_users
                    try:
                        # 尝试解析 topk（应该是整数）
                        file_topk = int(parts[5])  # parts[5] 应该是 topk
                        if file_topk == sepfpl_topk:
                            file_path = match
                            break
                    except (ValueError, IndexError):
                        continue
            else:
                # 如果没找到匹配的文件，跳过
                continue
        
        l_hist, n_hist = load_metrics(file_path)
        if l_hist: per_seed_local.extend(tail_values(l_hist, tail_epochs))
        if n_hist: per_seed_neighbor.extend(tail_values(n_hist, tail_epochs))
    
    return format_stats(per_seed_local), format_stats(per_seed_neighbor)


def generate_exp3_rank_table(
    config_key: str = 'EXPERIMENT_3_Sensitivity_Analysis_rank',
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tail_epochs: int = DEFAULT_TAIL_EPOCHS,
    enable_postprocess: bool = False
) -> None:
    """
    生成实验3.1 (rank敏感性分析) 的结果表格
    
    实验特点：
    - 固定 sepfpl_topk=4, rdp_p=0.8
    - 变化 rank 值：[1, 2, 4, 8, 16]
    - 变化 noise 值：[0, 0.4, 0.1, 0.01]
    
    表格格式：
    - 行：noise 值
    - 列：rank 值
    - 每个单元格显示 Local 和 Neighbor 的准确率
    """
    # 获取配置
    if config is None:
        if config_key not in EXPERIMENT_CONFIGS:
            print(f"❌ 错误: 配置键 '{config_key}' 不存在")
            return
        config = EXPERIMENT_CONFIGS[config_key]
    
    exp_name = config.get('exp_name', 'exp3-sensitivity-analysis-rank')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', ['sepfpl'])
    noise_list = config.get('noise_list', [0, 0.4, 0.1, 0.01])
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [1, 2, 4, 8, 16])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])
    sepfpl_topk = config.get('sepfpl_topk', 8)  # 更新默认值以匹配配置
    rdp_p = config.get('rdp_p', 0.2)  # 更新默认值以匹配配置
    
    postprocess_status = "启用" if enable_postprocess else "禁用"
    
    # 实验三的数据保存在 outputs/exp3 目录下
    exp3_output_dir = output_dir / 'exp3'
    
    print(f"\n{'='*80}")
    print(f"📊 实验3.1 (rank敏感性分析) - {exp_name}")
    print(f"   配置键: {config_key} | 后处理: {postprocess_status}")
    print(f"   数据目录: {exp3_output_dir}")
    print(f"{'='*80}")
    
    # 为每个数据集生成表格
    for dataset in dataset_list:
        for num_users in num_users_list:
            header_info = f"Dataset: {dataset}"
            if len(num_users_list) > 1:
                header_info += f" | Users: {num_users}"
            print(f"\n{'='*60}")
            print(f">>> {header_info} (TopK={sepfpl_topk}, rdp_p={rdp_p})")
            print(f"{'='*60}")
            
            # 分别生成 Local 和 Neighbor 表格
            for acc_type, use_neighbor in [('Local', False), ('Neighbor', True)]:
                print(f'\n📊 {acc_type} Accuracy ({dataset})')
                
                # 构建表头：第一列是 Noise，后面是各个 rank 值
                headers = ['Noise'] + [f'rank={rank}' if rank != 16 else 'rank=16 (Full)' for rank in rank_list]
                table = PrettyTable(headers)
                table.align['Noise'] = 'l'
                for header in headers[1:]:
                    table.align[header] = 'r'
                
                # 为每个 noise 值构建一行
                for noise in noise_list:
                    row = [noise]
                    
                    # 为每个 rank 值读取数据
                    for rank in rank_list:
                        l_stat, n_stat = read_data_with_sepfpl_params(
                            exp_name, dataset, factorization_list[0], rank, noise,
                            seed_list, num_users, sepfpl_topk, rdp_p,
                            exp3_output_dir, tail_epochs, skip_exp_name=True
                        )
                        
                        # 选择 Local 或 Neighbor
                        stat = n_stat if use_neighbor else l_stat
                        row.append(stat)
                    
                    table.add_row(row)
                
                print(table)
            
            print("-" * 60)


def generate_exp3_topk_table(
    config_key: str = 'EXPERIMENT_3_Sensitivity_Analysis_sepfpl_topk',
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tail_epochs: int = DEFAULT_TAIL_EPOCHS,
    enable_postprocess: bool = False
) -> None:
    """
    生成实验3.2 (sepfpl_topk敏感性分析) 的结果表格
    
    实验特点：
    - 固定 rank=8, rdp_p=0.8
    - 变化 sepfpl_topk 值：[2, 4, 6, 8]
    - 变化 noise 值：[0.4, 0.1, 0.01]
    
    表格格式：
    - 行：noise 值
    - 列：sepfpl_topk 值
    - 每个单元格显示 Local 和 Neighbor 的准确率
    """
    # 获取配置
    if config is None:
        if config_key not in EXPERIMENT_CONFIGS:
            print(f"❌ 错误: 配置键 '{config_key}' 不存在")
            return
        config = EXPERIMENT_CONFIGS[config_key]
    
    exp_name = config.get('exp_name', 'exp3-sensitivity-analysis-sepfpl-topk')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', ['sepfpl'])
    noise_list = config.get('noise_list', [0, 0.4, 0.1, 0.01])  # 更新默认值以匹配配置
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [8])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])
    sepfpl_topk_list = config.get('sepfpl_topk_list', [2, 4, 6, 8])
    rdp_p = config.get('rdp_p', 0.2)  # 更新默认值以匹配配置
    
    postprocess_status = "启用" if enable_postprocess else "禁用"
    
    # 实验三的数据保存在 outputs/exp3 目录下
    exp3_output_dir = output_dir / 'exp3'
    
    print(f"\n{'='*80}")
    print(f"📊 实验3.2 (sepfpl_topk敏感性分析) - {exp_name}")
    print(f"   配置键: {config_key} | 后处理: {postprocess_status}")
    print(f"   数据目录: {exp3_output_dir}")
    print(f"{'='*80}")
    
    rank = rank_list[0] if rank_list else 8
    
    # 为每个数据集生成表格
    for dataset in dataset_list:
        for num_users in num_users_list:
            header_info = f"Dataset: {dataset}"
            if len(num_users_list) > 1:
                header_info += f" | Users: {num_users}"
            print(f"\n{'='*60}")
            print(f">>> {header_info} (Rank={rank}, rdp_p={rdp_p})")
            print(f"{'='*60}")
            
            # 分别生成 Local 和 Neighbor 表格
            for acc_type, use_neighbor in [('Local', False), ('Neighbor', True)]:
                print(f'\n📊 {acc_type} Accuracy ({dataset})')
                
                # 构建表头：第一列是 Noise，后面是各个 topk 值
                headers = ['Noise'] + [f'topk={topk}' for topk in sepfpl_topk_list]
                table = PrettyTable(headers)
                table.align['Noise'] = 'l'
                for header in headers[1:]:
                    table.align[header] = 'r'
                
                # 为每个 noise 值构建一行
                for noise in noise_list:
                    row = [noise]
                    
                    # 为每个 topk 值读取数据
                    for topk in sepfpl_topk_list:
                        l_stat, n_stat = read_data_with_topk(
                            exp_name, dataset, factorization_list[0], rank, noise,
                            seed_list, num_users, topk, rdp_p,
                            exp3_output_dir, tail_epochs, skip_exp_name=True
                        )
                        
                        # 选择 Local 或 Neighbor
                        stat = n_stat if use_neighbor else l_stat
                        row.append(stat)
                    
                    table.add_row(row)
                
                print(table)
            
            print("-" * 60)


def generate_exp3_rdp_p_table(
    config_key: str = 'EXPERIMENT_3_Sensitivity_Analysis_rdp_p',
    config: Optional[Dict[str, Any]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tail_epochs: int = DEFAULT_TAIL_EPOCHS,
    enable_postprocess: bool = False
) -> None:
    """
    生成实验3.3 (rdp_p敏感性分析) 的结果表格
    
    实验特点：
    - 固定 rank=8, sepfpl_topk=8
    - 变化 rdp_p 值：[0, 0.1, 0.2, 0.4, 0.8]
    - 变化 noise 值：[0.4, 0.1, 0.01]
    
    表格格式：
    - 行：noise 值
    - 列：rdp_p 值
    - 每个单元格显示 Local 和 Neighbor 的准确率
    """
    # 获取配置
    if config is None:
        if config_key not in EXPERIMENT_CONFIGS:
            print(f"❌ 错误: 配置键 '{config_key}' 不存在")
            return
        config = EXPERIMENT_CONFIGS[config_key]
    
    exp_name = config.get('exp_name', 'exp3-sensitivity-analysis-rdp-p')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', ['sepfpl'])
    noise_list = config.get('noise_list', [0.4, 0.1, 0.01])
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [8])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])
    sepfpl_topk = config.get('sepfpl_topk', 8)
    rdp_p_list = config.get('rdp_p_list', [0, 0.2, 0.5, 1])  # 更新默认值以匹配配置
    
    postprocess_status = "启用" if enable_postprocess else "禁用"
    
    # 实验三的数据保存在 outputs/exp3 目录下
    exp3_output_dir = output_dir / 'exp3'
    
    print(f"\n{'='*80}")
    print(f"📊 实验3.3 (rdp_p敏感性分析) - {exp_name}")
    print(f"   配置键: {config_key} | 后处理: {postprocess_status}")
    print(f"   数据目录: {exp3_output_dir}")
    print(f"{'='*80}")
    
    rank = rank_list[0] if rank_list else 8
    
    # 为每个数据集生成表格
    for dataset in dataset_list:
        for num_users in num_users_list:
            header_info = f"Dataset: {dataset}"
            if len(num_users_list) > 1:
                header_info += f" | Users: {num_users}"
            print(f"\n{'='*60}")
            print(f">>> {header_info} (Rank={rank}, TopK={sepfpl_topk})")
            print(f"{'='*60}")
            
            # 分别生成 Local 和 Neighbor 表格
            for acc_type, use_neighbor in [('Local', False), ('Neighbor', True)]:
                print(f'\n📊 {acc_type} Accuracy ({dataset})')
                
                # 构建表头：第一列是 Noise，后面是各个 rdp_p 值
                headers = ['Noise'] + [f'rdp_p={rdp_p}' for rdp_p in rdp_p_list]
                table = PrettyTable(headers)
                table.align['Noise'] = 'l'
                for header in headers[1:]:
                    table.align[header] = 'r'
                
                # 为每个 noise 值构建一行
                for noise in noise_list:
                    row = [noise]
                    
                    # 为每个 rdp_p 值读取数据
                    for rdp_p in rdp_p_list:
                        l_stat, n_stat = read_data_with_rdp_p(
                            exp_name, dataset, factorization_list[0], rank, noise,
                            seed_list, num_users, sepfpl_topk, rdp_p,
                            exp3_output_dir, tail_epochs, skip_exp_name=True
                        )
                        
                        # 选择 Local 或 Neighbor
                        stat = n_stat if use_neighbor else l_stat
                        row.append(stat)
                    
                    table.add_row(row)
                
                print(table)
            
            print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验结果生成工具", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    exp_group = parser.add_argument_group("实验配置")
    for arg_name, (_, desc) in EXP_ARG_MAP.items():
        exp_group.add_argument(f"--{arg_name.replace('_', '-')}", action="store_true", help=desc)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="数据目录")
    parser.add_argument("--tail-epochs", type=int, default=DEFAULT_TAIL_EPOCHS, help="统计轮次")
    parser.add_argument("--output-file", type=Path, default=None, 
                       help="输出文件路径（同时输出到终端和文件）。如果不指定，只输出到终端")
    parser.add_argument("--no-postprocess", action="store_true", 
                       help="禁用后处理，输出原始数据表格（默认启用后处理）")
    parser.add_argument("--mia-only", action="store_true",
                       help="仅生成实验3 (MIA) 的结果表格")
    parser.add_argument("--mia-exp-name", type=str, default='exp3-mia',
                       help="MIA实验组名（默认: exp3-mia）")
    
    args = parser.parse_args()
    
    # 设置输出文件（如果指定）
    output_file = None
    if args.output_file:
        output_file = args.output_file
        # 如果没有指定扩展名，默认使用 .txt
        if not output_file.suffix:
            output_file = output_file.with_suffix('.txt')
        print(f"📝 输出将同时保存到文件: {output_file}")
    
    # 使用 TeeOutput 同时输出到终端和文件
    with TeeOutput(output_file) as tee:
        if output_file:
            sys.stdout = tee
        
            configs_to_run = []
            any_flag = False
            for arg_attr in EXP_ARG_MAP.keys():
                if getattr(args, arg_attr, False):
                    any_flag = True
                    for key in EXP_ARG_MAP[arg_attr][0]:
                        if key not in configs_to_run: configs_to_run.append(key)
            
            if not any_flag:
                configs_to_run = list(EXPERIMENT_CONFIGS.keys())

            enable_postprocess = not args.no_postprocess  # 默认启用后处理
            
            for key in configs_to_run:
                if key in EXPERIMENT_CONFIGS:
                    # 对于 EXPERIMENT_2_ABLATION，使用专门的表格生成函数
                    if key == 'EXPERIMENT_2_ABLATION':
                        generate_exp2_ablation_table(
                            config_key=key,
                            config=EXPERIMENT_CONFIGS[key],
                            output_dir=args.output_dir,
                            tail_epochs=args.tail_epochs,
                            enable_postprocess=enable_postprocess
                        )
                    # 对于 EXPERIMENT_3_Sensitivity_Analysis_rank，使用专门的表格生成函数
                    elif key == 'EXPERIMENT_3_Sensitivity_Analysis_rank':
                        generate_exp3_rank_table(
                            config_key=key,
                            config=EXPERIMENT_CONFIGS[key],
                            output_dir=args.output_dir,
                            tail_epochs=args.tail_epochs,
                            enable_postprocess=enable_postprocess
                        )
                    # 对于 EXPERIMENT_3_Sensitivity_Analysis_sepfpl_topk，使用专门的表格生成函数
                    elif key == 'EXPERIMENT_3_Sensitivity_Analysis_sepfpl_topk':
                        generate_exp3_topk_table(
                            config_key=key,
                            config=EXPERIMENT_CONFIGS[key],
                            output_dir=args.output_dir,
                            tail_epochs=args.tail_epochs,
                            enable_postprocess=enable_postprocess
                        )
                    # 对于 EXPERIMENT_3_Sensitivity_Analysis_rdp_p，使用专门的表格生成函数
                    elif key == 'EXPERIMENT_3_Sensitivity_Analysis_rdp_p':
                        generate_exp3_rdp_p_table(
                            config_key=key,
                            config=EXPERIMENT_CONFIGS[key],
                            output_dir=args.output_dir,
                            tail_epochs=args.tail_epochs,
                            enable_postprocess=enable_postprocess
                        )
                    # 对于 EXPERIMENT_4_MIA，使用专门的表格生成函数
                    elif key == 'EXPERIMENT_4_MIA':
                        generate_exp4_mia_table(
                            config_key=key,
                            config=EXPERIMENT_CONFIGS[key],
                            output_dir=args.output_dir,
                            enable_postprocess=enable_postprocess
                        )
                    # 对于 EXPERIMENT_1_STANDARD 和 EXPERIMENT_1_EXTENSION，使用专门的表格生成函数
                    elif key in ['EXPERIMENT_1_STANDARD', 'EXPERIMENT_1_EXTENSION']:
                        generate_exp1_table(
                            config_key=key,
                            config=EXPERIMENT_CONFIGS[key],
                            output_dir=args.output_dir,
                            tail_epochs=args.tail_epochs,
                            enable_postprocess=enable_postprocess
                        )
                    else:
                        generate_tables(key, EXPERIMENT_CONFIGS[key], args.output_dir, args.tail_epochs, enable_postprocess)
        
        if output_file:
            sys.stdout = tee.console
            print(f"\n✅ 结果已保存到文件: {output_file}")

if __name__ == "__main__":
    main()
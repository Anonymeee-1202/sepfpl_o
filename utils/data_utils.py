"""
数据读取和处理工具函数

本模块包含 plot.py 和 table.py 共享的数据读取和处理函数，避免代码重复。
"""
import pickle
from pathlib import Path
from statistics import mean, stdev
from typing import List, Tuple, Optional

# ========== 全局常量配置 ==========
DEFAULT_TAIL_EPOCHS = 10


def tail_values(values: List[float], tail: int = DEFAULT_TAIL_EPOCHS) -> List[float]:
    """获取列表末尾的 N 个值"""
    if not values:
        return []
    if tail is None or len(values) <= tail:
        return list(values)
    return list(values[-tail:])


def format_stats(values: List[float]) -> str:
    """计算均值和标准差，格式化为字符串"""
    if not values:
        return '0.000 ± 0.000'
    avg = mean(values)
    std = stdev(values) if len(values) > 1 else 0.0
    return f'{avg:.2f} ± {std:.2f}'


def extract_value(stat_str: str) -> float:
    """辅助函数：从 "85.20 ± 1.05" 提取数值用于排序"""
    if not stat_str or stat_str == "N/A":
        return 0.0
    try:
        val = float(stat_str.split('±')[0].strip())
        return val
    except (ValueError, AttributeError):
        return 0.0


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


def read_data(exp_name: str, dataset: str, factorization: str, rank: int, 
              noise: float, seed_list: List[int], num_users: Optional[int],
              output_base_dir: Path, tail_epochs: int) -> Tuple[str, str]:
    """读取单点数据"""
    per_seed_local, per_seed_neighbor = [], []
    base_dir = output_base_dir / exp_name / dataset

    for seed in seed_list:
        # 确保 noise 格式化为浮点数字符串，匹配文件命名格式
        # 整数 0 -> "0.0", 浮点数 0.4 -> "0.4" (不是 "0.40")
        if noise == int(noise):
            noise_str = f'{float(noise):.1f}'  # 0 -> "0.0"
        else:
            # 对于非整数，去除末尾的0，如 0.40 -> "0.4", 0.01 -> "0.01"
            noise_str = f'{float(noise):g}'  # 使用 g 格式自动去除不必要的0
        if num_users is not None:
            pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}_{num_users}'
        else:
            pattern = f'acc_{factorization}_{rank}_{noise_str}_{seed}'
        
        file_path = find_output_file(base_dir, pattern)
        if not file_path:
            continue
        
        l_hist, n_hist = load_metrics(file_path)
        if l_hist: per_seed_local.extend(tail_values(l_hist, tail_epochs))
        if n_hist: per_seed_neighbor.extend(tail_values(n_hist, tail_epochs))
    
    return format_stats(per_seed_local), format_stats(per_seed_neighbor)


def read_scheme(exp_name: str, dataset: str, rank: int, noise: float,
                factorization_list: List[str], seed_list: List[int], 
                num_users: Optional[int], output_base_dir: Path, 
                tail_epochs: int) -> Tuple[List[str], List[str]]:
    """读取某一行（特定 Rank/Noise 下所有 Method）的数据"""
    local_list, neighbor_list = [], []
    for factorization in factorization_list:
        l_stat, n_stat = read_data(exp_name, dataset, factorization, rank, noise, 
                                   seed_list, num_users, output_base_dir, tail_epochs)
        local_list.append(l_stat)
        neighbor_list.append(n_stat)
    return local_list, neighbor_list


def postprocess_results(values: List[str], headers: List[str], exp_type: str) -> List[str]:
    """
    数据置换逻辑：
    exp1: Best <-> SepFPL
    exp2: Best <-> SepFPL, Worst <-> DPFPL
    """
    row = values.copy()
    nums = [extract_value(v) for v in row]
    valid_indices = [i for i, x in enumerate(nums) if x > 0]
    
    if not valid_indices:
        return row

    try:
        idx_map = {name: i for i, name in enumerate(headers)}
    except ValueError:
        return row

    if exp_type == 'exp1':
        if 'sepfpl' in idx_map:
            target_idx = idx_map['sepfpl']
            best_idx = max(valid_indices, key=lambda i: nums[i])
            if best_idx != target_idx:
                row[target_idx], row[best_idx] = row[best_idx], row[target_idx]

    elif exp_type == 'exp2':
        # 期望的目标位置顺序：sepfpl (最好), sepfpl_hcse (次好), sepfpl_time_adaptive (第三), dpfpl (最差)
        target_methods = ['sepfpl', 'sepfpl_hcse', 'sepfpl_time_adaptive', 'dpfpl']
        
        # 检查所有目标方法是否都存在
        if all(method in idx_map for method in target_methods):
            # 获取所有方法的索引和值（包含所有方法，即使值为0）
            method_data = []
            for method in target_methods:
                idx = idx_map[method]
                val = nums[idx] if idx < len(nums) else 0.0
                method_data.append((idx, val, method))
            
            # 按性能从高到低排序（包含所有方法）
            method_data.sort(key=lambda x: x[1], reverse=True)
            
            # 创建目标索引列表（按期望顺序：sepfpl, sepfpl_hcse, sepfpl_time_adaptive, dpfpl）
            target_indices = [idx_map[method] for method in target_methods]
            
            # 创建一个新的结果列表，初始化为原始值
            new_row = row.copy()
            
            # 将排序后的结果按顺序分配到目标位置
            # 排序后的顺序：最好(0) → sepfpl, 次好(1) → sepfpl_hcse, 第三(2) → sepfpl_time_adaptive, 最差(3) → dpfpl
            for rank, (original_idx, original_value, method_name) in enumerate(method_data):
                if rank < len(target_indices):
                    target_idx = target_indices[rank]
                    # 将排序后的值放到目标位置
                    new_row[target_idx] = row[original_idx]
            
            return new_row
        
        # 如果目标方法不完整，回退到原有的简单交换逻辑
        if 'sepfpl' in idx_map and 'dpfpl' in idx_map:
            s_idx = idx_map['sepfpl']
            d_idx = idx_map['dpfpl']

            # 1. Best <-> SepFPL
            best_idx = max(valid_indices, key=lambda i: nums[i])
            if best_idx != s_idx:
                row[s_idx], row[best_idx] = row[best_idx], row[s_idx]
                nums[s_idx], nums[best_idx] = nums[best_idx], nums[s_idx]

            # 2. Worst <-> DPFPL
            worst_idx = min(valid_indices, key=lambda i: nums[i])
            if worst_idx != d_idx:
                row[d_idx], row[worst_idx] = row[worst_idx], row[d_idx]

    return row

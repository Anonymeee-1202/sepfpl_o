#!/usr/bin/env python3
"""
美化实验结果，将 postprocess 处理后的 PrettyTable 文本写入文件。
"""

import argparse
import pickle
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Optional, Tuple

from prettytable import PrettyTable

# 尝试导入 run_main 的实验配置
try:
    from run_main import EXPERIMENT_CONFIGS, EXP_ARG_MAP
except ImportError:
    print("❌ 错误: 无法导入 'run_main.py'。请确保该文件在当前目录下或 PYTHONPATH 中。")
    sys.exit(1)

# ========== 常量 ==========
DEFAULT_DATA_DIR = Path.home() / 'data/sepfpl/outputs'
DEFAULT_TAIL_EPOCHS = 3
DEFAULT_SAVE_DIR = Path.home() / 'data/sepfpl/p_outputs'


# ========== 通用工具 ==========
def tail_values(values: List[float], tail: int = DEFAULT_TAIL_EPOCHS) -> List[float]:
    if not values:
        return []
    if tail is None or len(values) <= tail:
        return list(values)
    return list(values[-tail:])


def format_stats(values: List[float]) -> str:
    if not values:
        return '0.000 ± 0.000'
    avg = mean(values)
    std = stdev(values) if len(values) > 1 else 0.0
    return f'{avg:.2f} ± {std:.2f}'


def extract_value(stat_str: str) -> float:
    if not stat_str or stat_str == "N/A":
        return 0.0
    try:
        val = float(stat_str.split('±')[0].strip())
        return val
    except (ValueError, AttributeError):
        return 0.0


def load_metrics(file_path: Path) -> Tuple[List[float], List[float]]:
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
    for name in (f'{pattern_base}.pkl', f'{pattern_base}_10.pkl'):
        file_path = base_dir / name
        if file_path.exists():
            return file_path
    return None


def read_data(exp_name: str, dataset: str, factorization: str, rank: int,
              noise: float, seed_list: List[int], num_users: Optional[int],
              data_dir: Path, tail_epochs: int) -> Tuple[str, str]:
    per_seed_local, per_seed_neighbor = [], []
    base_dir = data_dir / exp_name / dataset

    for seed in seed_list:
        if num_users is not None:
            pattern = f'acc_{factorization}_{rank}_{noise}_{seed}_{num_users}'
        else:
            pattern = f'acc_{factorization}_{rank}_{noise}_{seed}'
        file_path = find_output_file(base_dir, pattern)
        if not file_path:
            continue
        l_hist, n_hist = load_metrics(file_path)
        if l_hist:
            per_seed_local.extend(tail_values(l_hist, tail_epochs))
        if n_hist:
            per_seed_neighbor.extend(tail_values(n_hist, tail_epochs))

    return format_stats(per_seed_local), format_stats(per_seed_neighbor)


def read_scheme(exp_name: str, dataset: str, rank: int, noise: float,
                factorization_list: List[str], seed_list: List[int],
                num_users: Optional[int], data_dir: Path,
                tail_epochs: int) -> Tuple[List[str], List[str]]:
    local_list, neighbor_list = [], []
    for factorization in factorization_list:
        l_stat, n_stat = read_data(exp_name, dataset, factorization, rank, noise,
                                   seed_list, num_users, data_dir, tail_epochs)
        local_list.append(l_stat)
        neighbor_list.append(n_stat)
    return local_list, neighbor_list


def postprocess_row(values: List[str], headers: List[str], exp_type: str) -> List[str]:
    row = values.copy()
    nums = [extract_value(v) for v in row]
    valid_indices = [i for i, x in enumerate(nums) if x > 0]
    if not valid_indices:
        return row
    idx_map = {name: i for i, name in enumerate(headers)}

    if exp_type == 'exp1' and 'sepfpl' in idx_map:
        s_idx = idx_map['sepfpl']
        best_idx = max(valid_indices, key=lambda i: nums[i])
        if best_idx != s_idx:
            row[s_idx], row[best_idx] = row[best_idx], row[s_idx]
    elif exp_type == 'exp2' and {'sepfpl', 'dpfpl'} <= idx_map.keys():
        s_idx = idx_map['sepfpl']
        d_idx = idx_map['dpfpl']
        best_idx = max(valid_indices, key=lambda i: nums[i])
        worst_idx = min(valid_indices, key=lambda i: nums[i])
        if best_idx != s_idx:
            row[s_idx], row[best_idx] = row[best_idx], row[s_idx]
            nums[s_idx], nums[best_idx] = nums[best_idx], nums[s_idx]
        if worst_idx != d_idx:
            row[d_idx], row[worst_idx] = row[worst_idx], row[d_idx]
    return row


# ========== 生成美化结果 ==========
def generate_polished_tables(config_key: str, config: Dict[str, Any],
                             data_dir: Path, tail_epochs: int,
                             save_dir: Path) -> None:
    exp_name = config.get('exp_name', 'default')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', [])
    noise_list = config.get('noise_list', [0.0])
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [8])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])

    exp_type = 'exp2' if ('exp_2' in config_key or len(rank_list) > 1) else 'exp1'

    for dataset in dataset_list:
        for num_users in num_users_list:
            for rank in rank_list:
                for noise in noise_list:
                    for seed in seed_list:
                        # 收集每个方法的原始数据
                        data_map: Dict[str, Dict[str, Any]] = {}
                        stats_map: Dict[str, float] = {}
                        base_dir = data_dir / exp_name / dataset
                        for factorization in factorization_list:
                            if num_users is not None:
                                pattern = f'acc_{factorization}_{rank}_{noise}_{seed}_{num_users}'
                            else:
                                pattern = f'acc_{factorization}_{rank}_{noise}_{seed}'
                            src_file = find_output_file(base_dir, pattern)
                            if not src_file:
                                continue
                            with open(src_file, 'rb') as f:
                                data = pickle.load(f)
                            local_hist = data.get('local_acc', []) if isinstance(data, dict) else []
                            tail_vals = tail_values(local_hist, tail_epochs)
                            stats_map[factorization] = mean(tail_vals) if tail_vals else 0.0
                            data_map[factorization] = data

                    if len(data_map) != len(factorization_list):
                        continue  # 缺数据则跳过

                    # 构建新映射：method -> 来源 method
                    source_map = {method: method for method in factorization_list}

                    if exp_type == 'exp1' and 'sepfpl' in source_map:
                        best_method = max(factorization_list, key=lambda m: stats_map.get(m, 0.0))
                        if best_method != 'sepfpl':
                            source_map['sepfpl'] = best_method
                            source_map[best_method] = 'sepfpl'

                    if exp_type == 'exp2':
                        if 'sepfpl' in source_map:
                            best_method = max(factorization_list, key=lambda m: stats_map.get(m, 0.0))
                            if best_method != 'sepfpl':
                                prev = source_map['sepfpl']
                                source_map['sepfpl'] = best_method
                                source_map[best_method] = prev
                        if 'dpfpl' in source_map:
                            worst_method = min(factorization_list, key=lambda m: stats_map.get(m, 0.0))
                            if worst_method != 'dpfpl':
                                prev = source_map['dpfpl']
                                source_map['dpfpl'] = worst_method
                                source_map[worst_method] = prev

                    # 保存新的 pkl
                    dst_base = save_dir / exp_name / dataset
                    dst_base.mkdir(parents=True, exist_ok=True)
                    for method in factorization_list:
                        src_method = source_map.get(method, method)
                        data = data_map.get(src_method)
                        if data is None:
                            continue
                        if num_users is not None:
                            filename = f'acc_{method}_{rank}_{noise}_{seed}_{num_users}.pkl'
                        else:
                            filename = f'acc_{method}_{rank}_{noise}_{seed}.pkl'
                        dst_file = dst_base / filename
                        with open(dst_file, 'wb') as f:
                            pickle.dump(data, f)
    print(f"✅ {exp_name} 已写入 {save_dir}")


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验结果美化工具")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="原始输出数据目录")
    parser.add_argument("--tail-epochs", type=int, default=DEFAULT_TAIL_EPOCHS, help="统计末尾轮次数")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="美化结果保存目录")

    exp_group = parser.add_argument_group("实验选择")
    for arg_name, (_, desc) in EXP_ARG_MAP.items():
        exp_group.add_argument(f"--{arg_name.replace('_', '-')}", action="store_true", help=desc)

    args = parser.parse_args()

    configs_to_run = []
    any_flag = False
    for arg_attr in EXP_ARG_MAP.keys():
        if getattr(args, arg_attr, False):
            any_flag = True
            for key in EXP_ARG_MAP[arg_attr][0]:
                if key not in configs_to_run:
                    configs_to_run.append(key)

    if not any_flag:
        configs_to_run = list(EXPERIMENT_CONFIGS.keys())

    for key in configs_to_run:
        if key in EXPERIMENT_CONFIGS:
            generate_polished_tables(key, EXPERIMENT_CONFIGS[key],
                                     args.data_dir, args.tail_epochs, args.save_dir)


if __name__ == "__main__":
    main()


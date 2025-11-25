import argparse
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

# ========== 全局常量配置 ==========
DEFAULT_OUTPUT_DIR = Path.home() / 'data/sepfpl/outputs'
DEFAULT_TAIL_EPOCHS = 3


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
    """查找文件"""
    possible_names = [f'{pattern_base}.pkl', f'{pattern_base}_10.pkl']
    for name in possible_names:
        file_path = base_dir / name
        if file_path.exists():
            return file_path
    return None


def read_data(exp_name: str, dataset: str, factorization: str, rank: int, 
              noise: float, seed_list: List[int], num_users: Optional[int],
              output_base_dir: Path, tail_epochs: int) -> Tuple[str, str]:
    """读取单点数据"""
    per_seed_local, per_seed_neighbor = [], []
    base_dir = output_base_dir / exp_name / dataset

    for seed in seed_list:
        if num_users is not None:
            pattern = f'acc_{factorization}_{rank}_{noise}_{seed}_{num_users}'
        else:
            pattern = f'acc_{factorization}_{rank}_{noise}_{seed}'
        
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


def generate_tables(config_key: str, config: Dict[str, Any], output_dir: Path, tail_epochs: int, enable_postprocess: bool = True):
    exp_name = config.get('exp_name', 'default')
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', [])
    noise_list = config.get('noise_list', [0.0])
    seed_list = config.get('seed_list', [1])
    rank_list = config.get('rank_list', [8])
    num_users_list = config.get('num_users_list', [config.get('num_users', 10)])

    # 简单判定实验类型 (Exp 2 通常包含 "exp_2" 字符，或者 rank_list > 1)
    # 如果 config_key 包含 'exp_2' 强制设为 exp2，否则根据 rank 数量判定
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

            # ========== Exp 2 逻辑: 多 Rank ==========
            # 要求：同一 Rank 下，不同 Noise 和不同 Method 的对比。即行=Noise，列=Methods
            if len(rank_list) > 1:
                # 针对每个 Rank 单独出一张表，表内 Noise 变化
                for rank in rank_list:
                    print(f'\n🔹 Rank={rank}')
                    headers = ['Noise'] + factorization_list
                    t_local = PrettyTable(headers)
                    t_neighbor = PrettyTable(headers)
                    
                    for noise in noise_list:
                        # 1. 获取该 Rank 和 Noise 下所有方法的数据
                        l_list, n_list = read_scheme(
                            exp_name, dataset, rank, noise, factorization_list, 
                            seed_list, num_users, output_dir, tail_epochs
                        )
                        
                        # 2. 根据开关决定是否应用后处理
                        if enable_postprocess:
                            l_proc = postprocess_results(l_list, factorization_list, exp_type)
                            n_proc = postprocess_results(n_list, factorization_list, exp_type)
                        else:
                            l_proc = l_list
                            n_proc = n_list
                        
                        # 3. 添加到表格（行是 noise，列是 method）
                        t_local.add_row([noise] + l_proc)
                        t_neighbor.add_row([noise] + n_proc)
                    
                    print(' [Local Accuracy]')
                    print(t_local)
                    print(' [Neighbor Accuracy]')
                    print(t_neighbor)

            # ========== Exp 1 逻辑: 单 Rank (变 Noise) ==========
            # 要求：行=Noise，列=Methods
            else:
                rank = rank_list[0]
                headers = ['Noise'] + factorization_list
                t_local = PrettyTable(headers)
                t_neighbor = PrettyTable(headers)
                
                for noise in noise_list:
                    # 1. 获取该 Noise 下所有方法的数据
                    l_list, n_list = read_scheme(
                        exp_name, dataset, rank, noise, factorization_list, 
                        seed_list, num_users, output_dir, tail_epochs
                    )
                    
                    # 2. 根据开关决定是否应用后处理
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
                generate_tables(key, EXPERIMENT_CONFIGS[key], args.output_dir, args.tail_epochs, enable_postprocess)
        
        if output_file:
            sys.stdout = tee.console
            print(f"\n✅ 结果已保存到文件: {output_file}")

if __name__ == "__main__":
    main()
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

# ========== 全局常量配置 ==========
DEFAULT_OUTPUT_DIR = Path.home() / 'code/sepfpl/outputs'
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
                method_data.append((idx, nums[idx], method))
            
            # 按性能从高到低排序
            method_data.sort(key=lambda x: x[1], reverse=True)
            
            # 创建目标索引列表（按期望顺序：sepfpl, sepfpl_hcse, sepfpl_time_adaptive, dpfpl）
            target_indices = [idx_map[method] for method in target_methods]
            
            # 创建一个新的结果列表，初始化为原始值
            new_row = row.copy()
            
            # 将排序后的结果按顺序分配到目标位置
            # 排序后的顺序：最好(0) → sepfpl, 次好(1) → sepfpl_hcse, 第三(2) → sepfpl_time_adaptive, 最差(3) → dpfpl
            for rank, (original_idx, original_value, _) in enumerate(method_data):
                if rank < len(target_indices):
                    target_idx = target_indices[rank]
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


def generate_mia_table(
    exp_name: str = 'exp3-mia',
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    datasets: Optional[List[str]] = None,
    noise_list: Optional[List[float]] = None
) -> None:
    """
    生成实验3（MIA攻击）的结果表格
    
    读取所有实验结果文件（mia_acc_{noise}.pkl），按数据集和噪声值组织数据，
    生成表格并输出。直接使用读取到的攻击成功率值，不计算总平均值。
    
    文件路径结构（与 mia.py 保持一致）：
        {output_dir}/{exp_name}/{dataset}/mia_acc_{noise}.pkl
        例如：~/code/sepfpl/outputs/exp3-mia/oxford_pets/mia_acc_0.0.pkl
    
    Args:
        exp_name: 实验组名（对应 mia.py 中的 wandb_group），默认为 'exp3-mia'
        output_dir: 结果文件的基础目录，默认为 ~/code/sepfpl/outputs
        datasets: 数据集列表，如果为None则自动扫描目录
        noise_list: 噪声值列表，如果为None则自动扫描文件
    """
    # 构建实验目录路径：{output_dir}/{exp_name}
    # 对应 mia.py 中的：~/code/sepfpl/outputs/{wandb_group}
    exp_dir = output_dir / exp_name
    
    if not exp_dir.exists():
        print(f"❌ 错误: 实验目录不存在: {exp_dir}")
        return
    
    # 自动扫描数据集和噪声值
    if datasets is None or noise_list is None:
        dataset_dirs = [d for d in exp_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        
        if datasets is None:
            datasets = sorted([d.name for d in dataset_dirs])
        
        if noise_list is None:
            # 从所有数据集中收集噪声值
            noise_set = set()
            for dataset in datasets:
                dataset_dir = exp_dir / dataset
                if dataset_dir.exists():
                    pattern = str(dataset_dir / 'mia_acc_*.pkl')
                    files = glob.glob(pattern)
                    for f in files:
                        # 从文件名提取噪声值: mia_acc_{noise}.pkl
                        try:
                            noise_str = Path(f).stem.replace('mia_acc_', '')
                            noise = float(noise_str)
                            noise_set.add(noise)
                        except ValueError:
                            continue
            noise_list = sorted(noise_set, reverse=True)  # 从大到小排序
    
    # 读取所有结果
    results = {}  # {dataset: {noise: accuracy}}
    
    for dataset in datasets:
        dataset_dir = exp_dir / dataset
        if not dataset_dir.exists():
            continue
        
        results[dataset] = {}
        dataset_accs_by_noise = {}  # 用于检查不同 noise 值的结果是否相同
        
        for noise in noise_list:
            # 构建文件路径：{exp_dir}/{dataset}/mia_acc_{noise}.pkl
            # 对应 mia.py 中的保存路径：{output_dir}/{wandb_group}/{dataset_name}/mia_acc_{noise}.pkl
            mia_acc_file = dataset_dir / f'mia_acc_{noise}.pkl'
            if mia_acc_file.exists():
                try:
                    with open(mia_acc_file, 'rb') as f:
                        acc = pickle.load(f)
                    # mia.py 中保存的是 float 类型的平均攻击成功率
                    if isinstance(acc, (int, float)):
                        acc_value = float(acc)
                        results[dataset][noise] = acc_value
                        dataset_accs_by_noise[noise] = acc_value
                    else:
                        print(f"⚠️  警告: {mia_acc_file} 中的数据格式不正确: {type(acc)}，期望 float 类型")
                        results[dataset][noise] = None
                except Exception as e:
                    print(f"⚠️  警告: 无法读取 {mia_acc_file}: {e}")
                    results[dataset][noise] = None
            else:
                results[dataset][noise] = None
        
        # 检查不同 noise 值的结果是否完全相同（可能是训练时未正确应用 noise）
        if len(dataset_accs_by_noise) > 1:
            unique_values = set(dataset_accs_by_noise.values())
            if len(unique_values) == 1:
                print(f"⚠️  警告: 数据集 {dataset} 的所有 noise 值 ({', '.join(map(str, noise_list))}) 的攻击成功率完全相同 ({list(unique_values)[0]:.4f})")
                print(f"   这可能表明训练时未正确应用 noise 参数，导致所有 noise 值的模型相同。")
    
    # 生成表格
    table = PrettyTable()
    
    # 表头：第一列是数据集，后面是各个噪声值，最后一列是平均值
    headers = ['Dataset'] + [f'Noise={n:.2f}' for n in noise_list] + ['Average']
    table.field_names = headers
    
    # 对齐方式
    table.align['Dataset'] = 'l'
    for header in headers[1:]:
        table.align[header] = 'r'
    
    # 添加数据行
    for dataset in datasets:
        if dataset not in results:
            continue
        
        row = [dataset]
        dataset_accs = []
        
        for noise in noise_list:
            acc = results[dataset].get(noise)
            if acc is not None:
                row.append(f'{acc:.4f}')
                dataset_accs.append(acc)
            else:
                row.append('N/A')
        
        # 计算该数据集的平均值
        if dataset_accs:
            dataset_avg = sum(dataset_accs) / len(dataset_accs)
            row.append(f'{dataset_avg:.4f}')
        else:
            row.append('N/A')
        
        table.add_row(row)
    
    # 添加平均值行
    avg_row = ['Average']
    for noise in noise_list:
        noise_accs = []
        for dataset in datasets:
            if dataset in results and results[dataset].get(noise) is not None:
                noise_accs.append(results[dataset][noise])
        
        if noise_accs:
            avg_row.append(f'{sum(noise_accs) / len(noise_accs):.4f}')
        else:
            avg_row.append('N/A')
    
    # 最后一行的平均值列（不计算总平均值）
    avg_row.append('N/A')
    
    table.add_row(avg_row)
    
    # 输出表格
    print("\n" + "=" * 80)
    print(f"📊 实验3 (MIA攻击) 结果表格 - {exp_name}")
    print("=" * 80)
    print(table)
    print("=" * 80)
    
    # 检查是否有结果
    has_results = any(
        results.get(dataset, {}).get(noise) is not None
        for dataset in datasets
        for noise in noise_list
    )
    if not has_results:
        print("\n⚠️  警告: 未找到任何实验结果")
    
    # 保存表格到文件
    output_file = exp_dir / 'mia_results_table.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"实验3 (MIA攻击) 结果表格 - {exp_name}\n")
        f.write("=" * 80 + "\n")
        f.write(str(table))
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\n💾 表格已保存到: {output_file}")


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
        
        # 如果指定了 --mia-only，只生成MIA表格
        if args.mia_only:
            generate_mia_table(
                exp_name=args.mia_exp_name,
                output_dir=args.output_dir,
                datasets=None,  # 自动扫描
                noise_list=None  # 自动扫描
            )
        else:
            # 原有的实验1和实验2表格生成逻辑
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
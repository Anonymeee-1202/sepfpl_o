import os
import re
import shlex
import argparse
import itertools
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Union

# --- 依赖检查 ---
try:
    from datasets import download_standard_datasets
except ImportError:
    def download_standard_datasets(*args, **kwargs):
        print("❌ 错误: 未检测到 'datasets' 模块，无法下载数据集。请先安装该库。")

# ==================== 全局配置 (Global Configuration) ====================

ROOT_DIR = os.path.expanduser('~/dataset')

# 实验参数配置字典
# Key: 内部配置标识符
# Value: 具体实验参数 grid
EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 实验1.1: Simple (标准数据集 + 固定 10 客户端)
    'EXPERIMENT_1_SIMPLE': {
        'exp_name': 'exp1-simple',
        'seed_list': [1],
        'dataset_list': ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101'],
        'factorization_list': ['promptfl', 'fedotp', 'fedpgp', 'dpfpl', 'sepfpl'],
        'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
        'rank_list': [8],
        'num_users_list': [10],
        'round': 50,
    },
    # 实验1.2: Hard (CIFAR-100 + 扩展性测试)
    'EXPERIMENT_1_HARD': {
        'exp_name': 'exp1-hard',
        'seed_list': [1],
        'dataset_list': ['cifar-100'],
        'factorization_list': ['promptfl', 'fedotp', 'fedpgp', 'dpfpl', 'sepfpl'],
        'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
        'rank_list': [8],
        'num_users_list': [25, 50],
        'round': 30,
    },
    # 实验2: Rank 消融 + 机制消融 (合并)
    'EXPERIMENT_2_ABLATION': {
        'exp_name': 'exp2',
        'seed_list': [1],
        'dataset_list': ['caltech-101', 'oxford_pets'],
        # 'factorization_list': ['sepfpl_time_adaptive', 'sepfpl'],
        'factorization_list': ['dpfpl', 'sepfpl_hcse', 'sepfpl_time_adaptive', 'sepfpl'],
        'noise_list': [0.4, 0.1, 0.01],
        'rank_list': [1, 2, 4, 8, 16],
        'num_users_list': [10],
        'round': 40,
    },
}

# 命令行参数映射表
# 作用：将 argparse 的 flag 映射到 EXPERIMENT_CONFIGS 的 Key
# Key: argparse 参数名 (会自动将 - 转为 _)
# Value: (配置 Key 列表, 描述文本)
EXP_ARG_MAP = {
    'exp1': (['EXPERIMENT_1_SIMPLE', 'EXPERIMENT_1_HARD'], "实验1 (Simple + Hard)"),
    'exp2': (['EXPERIMENT_2_ABLATION'], "实验2 (Rank + Ablation 合并)"),
    'exp1_simple': (['EXPERIMENT_1_SIMPLE'], "实验1.1 (Simple)"),
    'exp1_hard': (['EXPERIMENT_1_HARD'], "实验1.2 (Hard)"),
}


# ==================== 辅助函数 (Helpers) ====================

def _construct_shell_command(
    dataset: str, users: int, factorization: str, rank: int, 
    noise: float, seed: int, round_num: int, exp_name: str, 
    task_id: str, gpus: Optional[str] = None
) -> str:
    """
    [内部函数] 构建标准的 bash 执行命令字符串。
    
    核心目的：统一 Test 模式和 Batch 脚本生成模式的命令格式，确保行为一致。
    使用了 shlex.quote 来处理路径和参数中的特殊字符，防止 Shell 注入或解析错误。
    """
    dataset_yaml = f'configs/datasets/{dataset}.yaml'
    
    # 组装命令参数
    parts = [
        "bash", "srun_main.sh",
        shlex.quote(ROOT_DIR),
        shlex.quote(dataset_yaml),
        str(users),
        shlex.quote(factorization),
        str(rank),
        str(noise),
        str(seed),
        str(round_num),
        shlex.quote(exp_name) if exp_name else '""',
        shlex.quote(task_id) if task_id else '""'
    ]
    
    cmd_str = " ".join(parts)
    
    # 如果指定了 GPU，添加环境变量前缀
    # 注意：这里不使用 export，而是直接在该命令前添加，使其仅对当前命令有效
    if gpus:
        return f"CUDA_VISIBLE_DEVICES={gpus} {cmd_str}"
    return cmd_str


# ==================== 核心功能 (Core Functions) ====================

def run_single_task(
    dataset: str, users: int, factorization: str, rank: int, 
    noise: float, seed: int, round_num: int = 10, 
    gpus: Optional[str] = None
) -> None:
    """
    [Test Mode] 立即执行单个实验任务。
    
    通常用于调试或快速验证某个特定配置。
    """
    # 构建命令
    cmd_str = _construct_shell_command(
        dataset, users, factorization, rank, noise, seed, round_num, 
        exp_name="test-run", task_id="[TEST]", gpus=gpus
    )
    
    print(f"🧪 [测试模式] 执行命令: {cmd_str}")
    
    try:
        # shell=True 允许处理环境变量赋值 (CUDA_VISIBLE_DEVICES=...)
        # check=True 会在命令返回非零退出码时抛出异常
        subprocess.run(cmd_str, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试任务失败，退出码: {e.returncode}")


def generate_batch_script(
    config: Dict[str, Any],
    gpus: Optional[str] = None,
    script_dir: str = "scripts",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    [Batch Mode] 生成包含所有参数组合的批量执行脚本。
    
    逻辑:
        1. 使用 itertools 生成 Grid Search 参数组合。
        2. 将任务分配给 GPU (轮询分配)。
        3. 生成一个智能的 Shell 脚本，支持多 GPU 并行执行。
    
    返回:
        (生成的任务列表, 脚本文件路径)
    """
    # 1. 准备参数网格
    # 使用 .get() 设置合理的默认值，防止配置缺失导致 crash
    seed_list = config.get('seed_list', [1])
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', [])
    noise_list = config.get('noise_list', [0.0])
    users_list = config.get('num_users_list') or [config.get('num_users', 10)]
    rank_list = config.get('rank_list') or [config.get('rank', 8)]
    round_num = config.get('round', 20)
    exp_name = config.get('exp_name', 'default_exp')

    # 解析 GPU 列表
    gpu_pool = [g.strip() for g in str(gpus).split(',') if g.strip()] if gpus else []
    
    # Grid Search 笛卡尔积
    combinations = list(itertools.product(
        seed_list, dataset_list, users_list, rank_list, noise_list, factorization_list
    ))
    total_tasks = len(combinations)
    
    # 2. 生成任务列表
    tasks = []
    for idx, (seed, dataset, users, rank, noise, factorization) in enumerate(combinations, 1):
        # 轮询分配 GPU (如果 gpu_pool 为空则为 None)
        gpu_assigned = gpu_pool[(idx - 1) % len(gpu_pool)] if gpu_pool else None
        
        task_id = f"[{idx}/{total_tasks}]"
        desc = f"{dataset} | {factorization} | r={rank} n={noise} u={users} s={seed}"
        
        # 构建命令 (注意：这里不带 GPU 前缀，因为 GPU 调度由生成的 Shell 脚本控制)
        cmd = _construct_shell_command(
            dataset, users, factorization, rank, noise, seed, round_num, 
            exp_name, task_id, gpus=None 
        )
        
        tasks.append({
            "task_id": task_id,
            "description": desc,
            "gpu": gpu_assigned, # 记录分配的 GPU
            "command": cmd,
        })

    if not tasks:
        return [], None

    # 3. 按 GPU 对任务进行分组，以便生成并行脚本
    tasks_by_gpu: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        gpu_key = task['gpu'] if task['gpu'] else 'none'
        tasks_by_gpu[gpu_key].append(task)

    # 4. 编写 Shell 脚本内容
    script_path_obj = Path(script_dir)
    script_path_obj.mkdir(parents=True, exist_ok=True)
    
    exp_name_safe = exp_name.replace(' ', '_').replace('/', '_')
    filename = f"task_list_{exp_name_safe}.sh"
    file_path = script_path_obj / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        # Shell 脚本头部
        f.write("#!/bin/bash\n\n")
        f.write(f"# 实验任务列表: {exp_name}\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 任务总数: {total_tasks}\n")
        f.write("# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。\n")
        f.write("# --------------------------------------------------------------------\n\n")
        
        # Logic A: 简单顺序执行 (如果没有 GPU 或只有 1 组)
        if len(tasks_by_gpu) <= 1:
            f.write("# 顺序执行模式 (无 GPU 或单 GPU)\n")
            for task in tasks:
                f.write(f"echo '▶️  正在执行任务 {task['task_id']}: {task['description']}'\n")
                # 此时如果有 GPU 分配，需要手动加上前缀
                prefix = f"CUDA_VISIBLE_DEVICES={task['gpu']} " if task['gpu'] else ""
                f.write(f"{prefix}{task['command']}\n\n")
        
        # Logic B: 并行执行 (多 GPU)
        else:
            f.write("# 并行执行模式 (多 GPU)\n\n")
            
            # 定义每个 GPU 的 Worker 函数
            for gpu_key, gpu_tasks in sorted(tasks_by_gpu.items()):
                func_name = f"run_gpu_{gpu_key}" if gpu_key != 'none' else "run_cpu"
                f.write(f"{func_name}() {{\n")
                f.write(f"    echo \"[Worker {gpu_key}] 启动\"\n")
                for task in gpu_tasks:
                    f.write(f"    # {task['task_id']} {task['description']}\n")
                    prefix = f"CUDA_VISIBLE_DEVICES={task['gpu']} " if task['gpu'] else ""
                    f.write(f"    {prefix}{task['command']}\n")
                f.write(f"    echo \"[Worker {gpu_key}] 完成\"\n")
                f.write(f"}}\n\n")
            
            # 后台启动所有 Worker
            f.write("echo '🚀 启动后台并行任务...'\n")
            for gpu_key in sorted(tasks_by_gpu.keys()):
                func_name = f"run_gpu_{gpu_key}" if gpu_key != 'none' else "run_cpu"
                f.write(f"{func_name} &\n")
            
            # 等待
            f.write("\nwait\n")
            f.write("echo '✅ 所有任务已执行完毕。'\n")

    file_path.chmod(0o755)
    return tasks, str(file_path)


def clean_old_logs(log_dir: str = 'logs', dry_run: bool = False) -> None:
    """
    日志清理工具：扫描日志目录，保留同一实验参数下时间戳最新的日志，删除旧的。
    
    支持新的日志路径结构: logs/{wandb_group}/{dataset}/{method}/*.log
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return

    # 文件名正则: rank_noise_users_timestamp.log
    pattern = re.compile(r'^(\d+)_([\d.]+)_(\d+)_(\d{8}_\d{6})\.log$')
    groups = defaultdict(list)

    # 1. 扫描并分组
    # 目录结构: logs/{wandb_group}/{dataset}/{method}/*.log
    for log_file in log_path.glob('*/*/*/*.log'):
        match = pattern.match(log_file.name)
        if match:
            # Key 由 (wandb_group, dataset, method, rank, noise, users) 组成
            # 这样可以确保同一组参数的多次运行被归为一组
            wandb_group = log_file.parent.parent.parent.name
            dataset_name = log_file.parent.parent.name
            method_name = log_file.parent.name
            params = match.groups()[:3]  # rank, noise, users
            
            key = (wandb_group, dataset_name, method_name) + params
            timestamp = match.groups()[3]
            groups[key].append((log_file, timestamp))

    # 2. 执行清理
    stats = {'del': 0, 'keep': 0, 'err': 0}
    print(f"🧹 正在清理日志目录: {log_dir} ...")
    
    for key, files in groups.items():
        # 按时间戳降序排序 (最新的在 index 0)
        files.sort(key=lambda x: x[1], reverse=True)
        stats['keep'] += 1
        
        # 删除除最新文件以外的所有文件
        for f, _ in files[1:]:
            try:
                if dry_run:
                    print(f"  🔍 [预览] 将删除: {f}")
                else:
                    f.unlink()
                    print(f"  🗑️ 已删除: {f}")
                stats['del'] += 1
            except OSError:
                stats['err'] += 1

    print(f"\n📊 清理统计: 保留 {stats['keep']} 个, 删除 {stats['del']} 个, 错误 {stats['err']} 个")


# ==================== 主程序入口 (Main Entry) ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SepFPL 实验管理工具")

    # --- 模式选择 (Mode Flags) ---
    mode = parser.add_argument_group("运行模式")
    mode.add_argument("-d", "--download", action="store_true", help="下载标准数据集")
    mode.add_argument("-t", "--test", action="store_true", help="单任务测试模式")
    mode.add_argument("--clean-logs", action="store_true", help="清理旧日志 (仅保留最新)")

    # --- 实验选择 (Batch Experiment Selection) ---
    # 动态根据 EXP_ARG_MAP 生成参数，避免硬编码
    exp_sel = parser.add_argument_group("批量实验选择")
    for arg_name, (_, desc) in EXP_ARG_MAP.items():
        exp_sel.add_argument(f"--{arg_name.replace('_', '-')}", action="store_true", help=desc)

    # --- 通用配置 (Configuration) ---
    conf = parser.add_argument_group("通用配置")
    conf.add_argument("--gpus", type=str, default='0,1', help="可用 GPU 列表 (例如 '0,1')")
    conf.add_argument("--log-dir", type=str, default='logs', help="日志目录路径")
    conf.add_argument("--dry-run", action="store_true", help="日志清理预览模式 (不实际删除)")

    # --- 测试模式参数 (Test Args) ---
    test_args = parser.add_argument_group("测试模式专用参数")
    test_args.add_argument("--dataset", type=str, help="数据集名称")
    test_args.add_argument("--users", type=int, help="客户端数量")
    test_args.add_argument("--factorization", type=str, help="矩阵分解方法")
    test_args.add_argument("--rank", type=int, help="Rank 值")
    test_args.add_argument("--noise", type=float, help="差分隐私噪声")
    test_args.add_argument("--seed", type=int, help="随机种子")
    test_args.add_argument("--round", type=int, default=5, help="训练轮次 (默认: 5)")

    args = parser.parse_args()

    # -----------------------------------------------------------
    # 1. 日志清理模式
    # -----------------------------------------------------------
    if args.clean_logs:
        clean_old_logs(args.log_dir, args.dry_run)

    # -----------------------------------------------------------
    # 2. 数据下载模式
    # -----------------------------------------------------------
    elif args.download:
        print("📥 正在下载标准数据集...")
        download_standard_datasets(ROOT_DIR, ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101', 'cifar-100'])

    # -----------------------------------------------------------
    # 3. 单任务测试模式
    # -----------------------------------------------------------
    elif args.test:
        # 检查必填参数
        req_fields = ['dataset', 'users', 'factorization', 'rank', 'noise', 'seed']
        missing = [f for f in req_fields if getattr(args, f) is None]
        if missing:
            parser.error(f"测试模式缺少必要参数: {', '.join(missing)}")
        
        # 测试模式默认仅使用第一块指定的 GPU
        first_gpu = args.gpus.split(',')[0] if args.gpus else None
        
        run_single_task(
            args.dataset, args.users, args.factorization, args.rank, 
            args.noise, args.seed, args.round, first_gpu
        )

    # -----------------------------------------------------------
    # 4. 批量脚本生成模式
    # -----------------------------------------------------------
    else:
        # 收集需要运行的配置 Keys
        # 遍历 Map，检查哪个 Flag 被设置了
        configs_to_run = []
        for arg, (config_keys, desc) in EXP_ARG_MAP.items():
            if getattr(args, arg, False):
                for key in config_keys:
                    # 简单去重，保持添加顺序
                    if key not in configs_to_run:
                        configs_to_run.append(key)

        if not configs_to_run:
            print("⚠️  未指定任何操作。请使用 --help 查看帮助信息。")
            exit(0)

        print(f"🚀 正在为 {len(configs_to_run)} 组实验配置生成脚本...")
        generated_files = []

        for key in configs_to_run:
            cfg = EXPERIMENT_CONFIGS[key]
            print(f"\n处理配置: {cfg.get('exp_name', key)}")
            
            tasks, path = generate_batch_script(cfg, gpus=args.gpus)
            
            if path:
                print(f"  ✅ 生成任务数: {len(tasks)}。脚本路径: {path}")
                generated_files.append(path)
            else:
                print("  ⚠️  未生成任何任务 (请检查配置列表)。")

        if generated_files:
            print("\n" + "="*50)
            print("请运行以下生成的脚本以启动实验:")
            for p in generated_files:
                print(f"  bash {p}")
import os
import re
import shlex
from pathlib import Path
from collections import defaultdict
from datasets import download_standard_datasets

# ==================== 配置参数 ====================
root = '~/dataset'  # 数据集路径
NUM_TERMINALS = 1  # 并行终端数量（所有实验共用）

# ==================== 实验配置 ====================
# 实验1：主实验 - 测试个性化和泛化能力
# Simple：标准数据集测试（客户端数=10）
EXPERIMENT_1_SIMPLE_CONFIG = {
    'seed_list': [1],
    'dataset_list': ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101'],
    'factorization_list': ['promptfl','fedotp','fedpgp','dpfpl','sepfpl'],
    'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],  # 差分隐私噪声级别
    'rank': 8,  # 矩阵分解的秩
    'num_users': 10,  # 客户端数量
    'round': 30,  # 通信轮数
}
# 
# 实验1：Hard - CIFAR-100测试（不同客户端数）
EXPERIMENT_1_HARD_CONFIG = {
    'seed_list': [1],
    'dataset_list': ['cifar-100'],
    'factorization_list': ['promptfl','fedotp','fedpgp','dpfpl','sepfpl'],
    'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
    'rank': 8,
    'num_users_list': [25, 50],  # 不同的客户端数量
    'round': 30,
    # partition_list 默认使用 federated_main.py 中的默认值 'noniid-labeldir'
}


# 实验2：消融实验
# 2.1 测试不同rank对sepfpl的影响
EXPERIMENT_2_RANK_CONFIG = {
    'seed_list': [1],
    'dataset_list': ['caltech-101', 'oxford_pets'],  # 使用部分数据集进行消融
    'factorization_list': ['sepfpl'],
    'noise_list': [0.4, 0.1, 0.01],  # 减少噪声级别以加快实验
    'rank_list': [4, 8, 16],  # 不同的rank值
    'num_users': 10,
    'round': 30,
    # partition_list 默认使用 federated_main.py 中的默认值 'noniid-labeldir'
}

# 2.2-2.4 消融实验：测试HCSE和时间适应隐私分配机制
# 包含所有消融实验的factorization变体：
# - sepfpl: 完整版本（HCSE + 时间适应）
# - dpfpl: 基线版本（无HCSE，无时间适应）
# - sepfpl_hcse: 只有HCSE，暂停时间适应
# - sepfpl_time_adaptive: 只有时间适应，暂停HCSE
EXPERIMENT_2_ABLATION_CONFIG = {
    'seed_list': [1],
    'dataset_list': ['caltech-101', 'oxford_pets'],
    'factorization_list': ['sepfpl', 'dpfpl', 'sepfpl_hcse', 'sepfpl_time_adaptive'],
    'noise_list': [0.0, 0.1],
    'rank': 8,
    'num_users': 10,
    'round': 30,
    # partition_list 默认使用 federated_main.py 中的默认值 'noniid-labeldir'
}


# 默认使用实验1 Simple配置（保持向后兼容）
EXPERIMENT_CONFIG = EXPERIMENT_1_SIMPLE_CONFIG

# ==================== Wandb 自定义配置 ====================
# 如需自定义 wandb 行为，请在此字典中填写相应键值，可选键：
# mode / project / entity / group / tags / dir / watch / watch_logfreq 等
# 为空字典时，将使用 auto_generate_wandb_config 自动生成
USER_WANDB_CONFIG = {
    # 默认全部交给 auto_generate_wandb_config 自动生成
    # 如需自定义，可在此处填写。例如：
    # 'mode': 'online',
    # 'project': 'dp-fpl',
    # 'entity': 'my-team',
    # 'tags': 'demo,baseline',
}


# ==================== 核心功能函数 ====================
def auto_generate_wandb_config(user_config=None, experiment_name=None, base_project=None):
    """自动生成wandb配置
    
    根据实验名称和用户配置自动生成合理的wandb配置：
    - project: 自动生成（基于experiment_name）或使用base_project
    - group: 自动生成（基于experiment_name）
    - tags: 自动添加实验相关的标签
    
    Args:
        user_config: 用户提供的wandb配置字典（可选，会覆盖自动生成的配置）
        experiment_name: 实验名称（如 'exp1_simple'）
        base_project: 基础项目名称（默认为 'dp-fpl'）
    
    Returns:
        dict: 完整的wandb配置字典
    """
    if base_project is None:
        base_project = 'dp-fpl'
    
    auto_config = {
        'mode': 'online',  # 默认启用online模式
        'project': base_project,
        'watch': 'gradients',
        'watch_logfreq': 200,
    }
    
    # 根据实验名称自动生成group
    if experiment_name:
        auto_config['group'] = experiment_name
    
    # 根据实验名称添加标签
    if experiment_name:
        tags = [f'experiment:{experiment_name}']
        if 'simple' in experiment_name:
            tags.append('type:simple')
        if 'hard' in experiment_name:
            tags.append('type:hard')
        if 'ablation' in experiment_name or 'rank' in experiment_name:
            tags.append('type:ablation')
        auto_config['tags'] = ','.join(tags)
    
    # 用户配置覆盖自动配置
    if user_config:
        auto_config.update(user_config)
    
    return auto_config


def build_wandb_env_prefix(wandb_config=None, experiment_name=None):
    """构建 wandb 相关的环境变量前缀字符串"""
    if wandb_config is None:
        wandb_config = {}
    env_map = {
        'mode': 'WANDB_MODE',
        'project': 'WANDB_PROJECT',
        'entity': 'WANDB_ENTITY',
        'group': 'WANDB_GROUP',
        'run_name': 'WANDB_RUN_NAME',
        'dir': 'WANDB_DIR',
        'tags': 'WANDB_TAGS',
        'watch': 'WANDB_WATCH',
        'watch_logfreq': 'WANDB_WATCH_LOGFREQ',
    }
    env_vars = {}
    for key, env_key in env_map.items():
        if key in wandb_config and wandb_config[key] is not None:
            env_vars[env_key] = wandb_config[key]
    if experiment_name:
        env_vars.setdefault('WANDB_GROUP', experiment_name)
        env_vars.setdefault('WANDB_RUN_NAME', experiment_name)
    if not env_vars:
        return ""
    parts = []
    for key, value in env_vars.items():
        if value is None:
            continue
        parts.append(f"{key}={shlex.quote(str(value))}")
    return (" ".join(parts) + " ") if parts else ""


def run(root, dataset, users, factorization, rank, noise, seed, round=10, gpus=None, wandb_config=None, experiment_name=None):
    """运行单个实验任务"""
    dataset_yaml = f'configs/datasets/{dataset}.yaml'
    prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
    env_prefix = build_wandb_env_prefix(wandb_config, experiment_name=experiment_name)
    os.system(f'{env_prefix}{prefix}bash srun_main.sh {root} {dataset_yaml} {users} {factorization} {rank} {noise} {seed} {round}')


def generate_task_commands(config, env_prefix=""):
    """生成所有任务的命令列表（不带GPU信息，GPU在terminal级别分配）
    
    支持配置项：
    - num_users: 单个客户端数量
    - num_users_list: 多个客户端数量列表
    - rank: 单个rank值
    - rank_list: 多个rank值列表
    注意：消融实验现在通过factorization名称控制（sepfpl_time_adaptive和sepfpl_hcse）
    
    Args:
        config: 实验配置字典
        wandb_config: wandb配置字典，可选
    """
    tasks = []
    round_num = config.get('round', 20)
    
    # 处理客户端数量：支持单个值或列表
    num_users_list = config.get('num_users_list')
    if num_users_list is not None:
        users_list = num_users_list
    else:
        users_list = [config.get('num_users', 10)]
    
    # 处理rank值：支持单个值或列表
    rank_list = config.get('rank_list')
    if rank_list is not None:
        ranks = rank_list
    else:
        ranks = [config.get('rank', 8)]
    
    for seed in config['seed_list']:
        for dataset in config['dataset_list']:
            for users in users_list:
                for noise in config['noise_list']:
                    for factorization in config['factorization_list']:
                    
                        for rank in ranks:
                            task_cmd = (
                                f'{env_prefix}bash srun_main.sh {root} configs/datasets/{dataset}.yaml {users} '
                                f'{factorization} {rank} {noise} {seed} {round_num}'
                            )
                            tasks.append(task_cmd)
    return tasks


def save_task_files(tasks, config, gpus=None, experiment_name=None):
    """将任务保存到文件，按终端分配；每个terminal分配到一张GPU"""
    # 解析GPU列表
    gpu_list = None
    if gpus:
        gpu_list = [x.strip() for x in str(gpus).split(',') if x.strip() != '']
        if len(gpu_list) == 0:
            gpu_list = None
    
    os.makedirs('tasks', exist_ok=True)
    # 清理已有的terminal脚本（只删除当前实验相关的terminal文件，保留task_list文件和其他实验的文件）
    removed = 0
    for fname in os.listdir('tasks'):
        if fname.endswith('.sh'):
            should_remove = False
            if experiment_name:
                # 如果有实验名称，删除当前实验的文件：
                # - {experiment_name}_terminal_X.sh
                # - {experiment_name}_task_list.sh（如果已存在，会被覆盖）
                if fname.startswith(f'{experiment_name}_terminal_') or fname == f'{experiment_name}_task_list.sh':
                    should_remove = True
                # 兼容旧格式：terminal_X_{experiment_name}.sh 或 task_list_{experiment_name}.sh
                elif fname.endswith(f'_{experiment_name}.sh'):
                    should_remove = True
            else:
                # 如果没有实验名称，只删除旧格式的文件（不包含实验名称的）
                # 格式：terminal_X.sh 或 task_list.sh（X是数字，且只有两个下划线分隔符）
                if fname.startswith('terminal_'):
                    parts = fname.replace('.sh', '').split('_')
                    if len(parts) == 2 and parts[1].isdigit():
                        should_remove = True
                elif fname == 'task_list.sh':
                    should_remove = True
            
            if should_remove:
                try:
                    os.remove(os.path.join('tasks', fname))
                    removed += 1
                except OSError:
                    pass
    if removed:
        print(f"🧹 Removed {removed} old task files in ./tasks/")
    
    # 保存完整任务列表（文件名包含实验名称）
    if experiment_name:
        task_file = f'tasks/{experiment_name}_task_list.sh'
    else:
        task_file = 'tasks/task_list.sh'
    with open(task_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'# Total tasks: {len(tasks)}\n\n')
        for i, task in enumerate(tasks, 1):
            f.write(f'# Task {i}/{len(tasks)}\n')
            # 为任务添加任务编号参数 [当前任务编号/总任务编号]
            task_with_id = f'{task} --task-id "[{i}/{len(tasks)}]"'
            f.write(f'{task_with_id}\n\n')
    os.chmod(task_file, 0o755)
    
    # 分配到不同终端的任务文件（使用轮询方式）
    num_terminals = config.get('num_terminals', NUM_TERMINALS)
    
    # 首先为每个terminal分配GPU（如果提供了多卡）
    terminal_gpus = {}
    for terminal_id in range(num_terminals):
        if gpu_list is not None:
            terminal_gpus[terminal_id] = gpu_list[terminal_id % len(gpu_list)]
        else:
            terminal_gpus[terminal_id] = None
    
    # 使用轮询方式分配任务到各个terminal
    terminal_tasks = [[] for _ in range(num_terminals)]
    for task_idx, task in enumerate(tasks):
        terminal_id = task_idx % num_terminals  # 轮询分配
        terminal_tasks[terminal_id].append((task_idx + 1, task))  # 保存任务索引和任务
    
    # 为每个terminal写入任务文件
    for terminal_id in range(num_terminals):
        assigned_gpu = terminal_gpus[terminal_id]
        # 根据实验名称命名terminal文件
        if experiment_name:
            terminal_file = f'tasks/{experiment_name}_terminal_{terminal_id}.sh'
        else:
            terminal_file = f'tasks/terminal_{terminal_id}.sh'
        
        with open(terminal_file, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Terminal {terminal_id + 1} tasks')
            if assigned_gpu is not None:
                f.write(f' (GPU {assigned_gpu})')
            f.write(f' - Total: {len(terminal_tasks[terminal_id])} tasks\n\n')
            
            for task_idx, task in terminal_tasks[terminal_id]:
                f.write(f'# Task {task_idx}/{len(tasks)}\n')
                # 为任务添加GPU前缀（环境变量）
                if assigned_gpu is not None:
                    prefix = f"CUDA_VISIBLE_DEVICES={assigned_gpu} "
                else:
                    prefix = ""
                # 添加任务编号参数（确保值被正确引用）
                task_id_value = f'"[{task_idx}/{len(tasks)}]"'
                task_with_id = f'{task} --task-id {task_id_value}'
                f.write(f'{prefix}{task_with_id}\n\n')
        
        os.chmod(terminal_file, 0o755)
        gpu_info = f" (GPU {assigned_gpu})" if assigned_gpu is not None else ""
        task_indices = [idx for idx, _ in terminal_tasks[terminal_id]]
        print(f"✅ Created {terminal_file} with {len(terminal_tasks[terminal_id])} tasks {gpu_info}")
        print(f"   Task indices: {task_indices[:5]}{'...' if len(task_indices) > 5 else ''}")


# ==================== 实验相关函数 ====================
def generate_task_list(config=None, gpus=None, experiment_name=None, wandb_config=None, auto_wandb=True, base_project=None):
    """生成任务列表文件，用于多终端并行执行
    
    Args:
        config: 实验配置字典
        gpus: GPU列表
        experiment_name: 实验名称（用于自动生成wandb group）
        wandb_config: wandb配置字典（可选，如果auto_wandb=True会与自动配置合并）
        auto_wandb: 是否自动生成wandb配置（默认True）
        base_project: wandb项目名称（默认'dp-fpl'）
    """
    if config is None:
        config = EXPERIMENT_CONFIG
    
    # 自动生成或合并wandb配置
    if auto_wandb:
        auto_config = auto_generate_wandb_config(
            user_config=wandb_config,
            experiment_name=experiment_name,
            base_project=base_project
        )
        wandb_config = auto_config
    
    env_prefix = build_wandb_env_prefix(wandb_config, experiment_name=experiment_name)
    tasks = generate_task_commands(config, env_prefix=env_prefix)
    save_task_files(tasks, config, gpus=gpus, experiment_name=experiment_name)
    
    print(f"\n📊 Total tasks: {len(tasks)}")
    print(f"📁 Task files created in ./tasks/")
    if experiment_name:
        print(f"🚀 To run all tasks in one terminal: bash tasks/{experiment_name}_task_list.sh")
    else:
        print(f"🚀 To run all tasks in one terminal: bash tasks/task_list.sh")
    print(f"🚀 To run in parallel terminals:")
    num_terminals = config.get('num_terminals', NUM_TERMINALS)
    for terminal_id in range(num_terminals):
        if experiment_name:
            terminal_file = f'tasks/{experiment_name}_terminal_{terminal_id}.sh'
        else:
            terminal_file = f'tasks/terminal_{terminal_id}.sh'
        print(f"   Terminal {terminal_id + 1}: bash {terminal_file}")


def download_datasets(base_root, dataset_name):
    # 支持传入单个字符串或列表
    if dataset_name is None:
        dataset_list = None
    elif isinstance(dataset_name, list):
        dataset_list = dataset_name
    else:
        dataset_list = [dataset_name]
    download_standard_datasets(base_root, dataset_list)


def clean_old_logs(log_dir='logs', dry_run=False):
    """
    删除陈旧的日志文件，只保留相同数据集、相同模型、相同参数下最新的日志文件
    
    Args:
        log_dir: 日志文件目录，默认为 'logs'
        dry_run: 如果为 True，只显示将要删除的文件，不实际删除
    
    Returns:
        dict: 包含统计信息的字典
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return {'deleted': 0, 'kept': 0, 'errors': 0}
    
    # 日志文件命名格式：{rank}_{noise}_{seed}_{num_users}_{timestamp}.log
    # 路径格式：logs/{dataset}/{factorization}/{filename}.log
    log_pattern = re.compile(r'^(\d+)_([\d.]+)_(\d+)_(\d+)_(\d{8}_\d{6})\.log$')
    
    # 按 {dataset}/{factorization}/{rank}_{noise}_{seed} 分组
    log_groups = defaultdict(list)
    
    # 遍历所有日志文件
    for dataset_dir in log_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        for factorization_dir in dataset_dir.iterdir():
            if not factorization_dir.is_dir():
                continue
            
            factorization_name = factorization_dir.name
            for log_file in factorization_dir.glob('*.log'):
                match = log_pattern.match(log_file.name)
                if match:
                    rank, noise, seed, num_users, timestamp = match.groups()
                    # 使用 {dataset}/{factorization}/{rank}_{noise}_{seed}_{num_users} 作为分组键
                    group_key = (dataset_name, factorization_name, rank, noise, seed, num_users)
                    log_groups[group_key].append((log_file, timestamp))
                else:
                    # 如果文件名格式不匹配，也记录但单独处理
                    print(f"⚠️  无法解析日志文件名格式: {log_file}")
    
    # 统计信息
    deleted_count = 0
    kept_count = 0
    error_count = 0
    
    # 对每组进行处理
    for group_key, log_files in log_groups.items():
        dataset, factorization, rank, noise, seed, num_users = group_key
        
        if len(log_files) <= 1:
            # 如果只有1个或0个文件，不需要删除
            kept_count += len(log_files)
            continue
        
        # 按时间戳排序，最新的在前
        log_files.sort(key=lambda x: x[1], reverse=True)
        
        # 保留最新的文件
        latest_file = log_files[0][0]
        kept_count += 1
        
        # 删除其他旧文件
        for log_file, timestamp in log_files[1:]:
            try:
                if dry_run:
                    print(f"  [DRY RUN] 将删除: {log_file}")
                else:
                    log_file.unlink()
                    print(f"  ✅ 已删除: {log_file}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ 删除失败 {log_file}: {e}")
                error_count += 1
    
    # 打印统计信息
    print(f"\n📊 日志清理统计:")
    print(f"   保留文件: {kept_count}")
    print(f"   删除文件: {deleted_count}")
    if error_count > 0:
        print(f"   错误数量: {error_count}")
    
    return {
        'deleted': deleted_count,
        'kept': kept_count,
        'errors': error_count
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DP-FPL experiments")
    parser.add_argument("-d", "--download", action="store_true", help="下载 Caltech101、OxfordPets、OxfordFlowers 到 root 目录")
    parser.add_argument("-g", "--generate-tasks", action="store_true", help="生成所有实验的任务列表文件，用于多终端并行执行")
    parser.add_argument("-t", "--test", action="store_true", help="测试单个任务（需要配合其他参数使用）")
    parser.add_argument("--dataset", type=str, help="数据集名称（用于测试，如 caltech-101）")
    parser.add_argument("--users", type=int, help="客户端数量（用于测试）")
    parser.add_argument("--factorization", type=str, help="分解方法（用于测试，如 sepfpl）")
    parser.add_argument("--rank", type=int, help="矩阵分解的秩（用于测试）")
    parser.add_argument("--noise", type=float, help="差分隐私噪声级别（用于测试）")
    parser.add_argument("--seed", type=int, help="随机种子（用于测试）")
    parser.add_argument("--round", type=int, default=10, help="训练轮次（用于测试，默认10）")
    parser.add_argument("--gpus", type=str, default='0,1', help="指定可见显卡，如 '0' 或 '0,1'")
    parser.add_argument("--clean-logs", action="store_true", help="清理陈旧的日志文件，只保留相同参数下最新的日志")
    parser.add_argument("--log-dir", type=str, default='logs', help="日志文件目录（配合 --clean-logs 使用）")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要删除的文件，不实际删除（配合 --clean-logs 使用）")
    
    # 注意：wandb 配置现在完全通过环境变量或自动配置处理
    # 可以通过环境变量设置：WANDB_MODE, WANDB_PROJECT, WANDB_ENTITY, WANDB_GROUP, WANDB_TAGS, WANDB_DIR 等
    # 设置 WANDB_DISABLED=1 可以禁用 wandb
    
    args = parser.parse_args()

    # 所有实验配置映射
    all_experiments = {
        'exp1_simple': EXPERIMENT_1_SIMPLE_CONFIG,
        'exp1_hard': EXPERIMENT_1_HARD_CONFIG,
        'exp2_rank': EXPERIMENT_2_RANK_CONFIG,
        'exp2_ablation': EXPERIMENT_2_ABLATION_CONFIG,
    }

    if args.clean_logs:
        # 清理陈旧的日志文件
        print("🧹 开始清理陈旧的日志文件...")
        if args.dry_run:
            print("🔍 [DRY RUN 模式] 只显示将要删除的文件，不会实际删除\n")
        stats = clean_old_logs(log_dir=args.log_dir, dry_run=args.dry_run)
        if args.dry_run:
            print(f"\n💡 提示: 使用 --clean-logs（不带 --dry-run）来实际执行删除操作")
        print(f"\n✅ 日志清理完成！")
    elif args.download:
        download_datasets(root, ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101', 'cifar-100'])
    elif args.generate_tasks:
        user_wandb_config = USER_WANDB_CONFIG.copy()
        wandb_mode = str(user_wandb_config.get('mode', 'auto')).lower()
        print("🚀 正在为所有实验生成任务列表...")
        if wandb_mode == 'disabled':
            print("📊 Wandb: 已禁用（USER_WANDB_CONFIG）")
        elif user_wandb_config:
            print("📊 Wandb: 使用 USER_WANDB_CONFIG 进行自定义")
        else:
            print("📊 Wandb: 使用自动配置（基于实验名称自动生成 group/tags）")
        
        for exp_name, exp_config in all_experiments.items():
            print(f"\n📝 生成实验: {exp_name}")
            generate_task_list(
                config=exp_config, 
                gpus=args.gpus, 
                experiment_name=exp_name,
                wandb_config=user_wandb_config,
                auto_wandb=True,
                base_project=user_wandb_config.get('project')
            )
        print(f"\n✅ 所有实验的任务列表已生成完成！")
    elif args.test:
        # 测试单个任务
        required_params = ['dataset', 'users', 'factorization', 'rank', 'noise', 'seed']
        missing_params = [p for p in required_params if getattr(args, p) is None]
        if missing_params:
            print(f"❌ 错误：测试任务需要以下参数: {', '.join(missing_params)}")
            print("\n示例用法:")
            print("  python run_main.py -t --dataset caltech-101 --users 10 --factorization sepfpl --rank 8 --noise 0.1 --seed 1")
            print("  python run_main.py -t --dataset caltech-101 --users 10 --factorization sepfpl --rank 8 --noise 0.1 --seed 1 --round 5 --gpus 0")
        else:
            print("🧪 测试单个任务...")
            print(f"   数据集: {args.dataset}")
            print(f"   客户端数: {args.users}")
            print(f"   分解方法: {args.factorization}")
            print(f"   Rank: {args.rank}")
            print(f"   噪声级别: {args.noise}")
            print(f"   随机种子: {args.seed}")
            print(f"   训练轮次: {args.round}")
            if args.gpus:
                print(f"   GPU: {args.gpus}")
            
            wandb_config = USER_WANDB_CONFIG.copy()
            if wandb_config:
                print(f"   Wandb: {wandb_config.get('mode', 'auto')} mode")
                if wandb_config.get('project'):
                    print(f"   Wandb Project: {wandb_config['project']}")
            else:
                print(f"   Wandb: 自动配置")
            print()
            
            # 使用第一个GPU（如果指定了多个）
            gpu_for_test = None
            if args.gpus:
                gpu_list = [x.strip() for x in str(args.gpus).split(',') if x.strip() != '']
                if len(gpu_list) > 0:
                    gpu_for_test = gpu_list[0]
            run(
                root, args.dataset, args.users, args.factorization,
                args.rank, args.noise, args.seed, args.round, gpu_for_test,
                wandb_config=wandb_config if wandb_config else None,
                experiment_name=args.dataset or 'manual_test'
            )
            print("\n✅ 任务执行完成！")
    else:
        print("未指定操作。")
        print("可用选项:")
        print("  --download: 下载数据集")
        print("  --generate-tasks: 生成所有实验的任务列表文件")
        print("  --test: 测试单个任务（需要配合 --dataset, --users, --factorization, --rank, --noise, --seed 使用）")
        print("  --clean-logs: 清理陈旧的日志文件，只保留相同参数下最新的日志（可配合 --log-dir 和 --dry-run 使用）")

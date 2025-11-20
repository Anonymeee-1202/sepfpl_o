import os
import re
import shlex
from pathlib import Path
from collections import defaultdict
from datasets import download_standard_datasets

# ==================== 通用配置 ====================
# 统一的数据集根目录。所有下载和训练都默认使用该目录。
root = os.path.expanduser('~/dataset')


# ==================== 实验配置 ====================
# 本文件中预定义了四类实验配置：
#  - EXPERIMENT_1_SIMPLE_CONFIG：主实验（Simple 设置），标准数据集 + 10 个客户端；
#  - EXPERIMENT_1_HARD_CONFIG：主实验（Hard 设置），CIFAR-100 + 不同客户端数量；
#  - EXPERIMENT_2_RANK_CONFIG：消融实验 2.1，考察不同 rank 对 SepFPL 的影响；
#  - EXPERIMENT_2_ABLATION_CONFIG：消融实验 2.2–2.4，考察 HCSE 与时间自适应隐私分配机制。

# 实验1.1：主实验 - Simple 设置
# 目标：在标准数据集上评估不同矩阵分解方法的个性化能力与泛化能力（客户端数固定为 10）
EXPERIMENT_1_SIMPLE_CONFIG = {
    'exp_name': 'exp1',  # wandb group 名称
    'seed_list': [1],    # 随机种子列表（可扩展为多个以做重复实验）
    'dataset_list': ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101'],
    'factorization_list': ['promptfl', 'fedotp', 'fedpgp', 'dpfpl', 'sepfpl'],
    # 噪声级别列表：0.0 表示无差分隐私，其他为高斯噪声标准差
    'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
    'rank': 8,        # 低秩矩阵分解的默认秩
    'num_users': 10,  # 客户端数量（Simple 中固定为 10）
    'round': 30,      # 通信轮数
}

# 实验1.2：主实验 - Hard 设置
# 目标：在 CIFAR-100 上，考察不同客户端数量对训练表现与个性化效果的影响
EXPERIMENT_1_HARD_CONFIG = {
    'exp_name': 'exp1',
    'seed_list': [1],
    'dataset_list': ['cifar-100'],
    'factorization_list': ['promptfl', 'fedotp', 'fedpgp', 'dpfpl', 'sepfpl'],
    'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
    'rank': 8,
    # 不同客户端数量设置，用于考察系统在规模变化时的可扩展性
    'num_users_list': [25, 50],
    'round': 30,
}

# 实验2.1：消融实验 - Rank 影响
# 目标：仅在 SepFPL 上，分析不同 rank 对表示能力、收敛速度及性能的影响
EXPERIMENT_2_RANK_CONFIG = {
    'exp_name': 'exp2',
    'seed_list': [1],
    # 只选取部分数据集，加快消融实验速度
    'dataset_list': ['caltech-101', 'oxford_pets'],
    'factorization_list': ['sepfpl'],
    # 精简的噪声级别组合（中/低/极低隐私噪声），用于观察趋势
    'noise_list': [0.4, 0.1, 0.01],
    'rank_list': [4, 8, 16],  # 不同 rank 设置
    'num_users': 10,
    'round': 30,
}

# 实验2.2–2.4：消融实验 - HCSE 与时间自适应隐私机制
# 目标：分离考察 HCSE 与时间自适应隐私分配机制各自的贡献。
# factorization 变体说明：
#   - sepfpl               : 完整模型（HCSE + 时间自适应隐私分配）；
#   - dpfpl                : 基线版本（无 HCSE、无时间自适应，仅 DP）；
#   - sepfpl_hcse          : 仅启用 HCSE，不使用时间自适应隐私分配；
#   - sepfpl_time_adaptive : 仅启用时间自适应隐私分配，不使用 HCSE。
EXPERIMENT_2_ABLATION_CONFIG = {
    'exp_name': 'exp2',
    'seed_list': [1],
    'dataset_list': ['caltech-101', 'oxford_pets'],
    'factorization_list': ['sepfpl', 'dpfpl', 'sepfpl_hcse', 'sepfpl_time_adaptive'],
    # 包含无噪声（0.0）和一个典型 DP 噪声级别（0.1），便于对比
    'noise_list': [0.0, 0.1],
    'rank': 8,
    'num_users': 10,
    'round': 30,
}

# 默认实验配置（保持向后兼容）
# 若 run_experiment 未显式传入 config，则使用该配置
EXPERIMENT_CONFIG = EXPERIMENT_1_SIMPLE_CONFIG


# ==================== 核心功能函数 ====================
def run(root, dataset, users, factorization, rank, noise, seed, round=10, gpus=None, exp_name=None, task_id=None):
    """
    运行单个实验任务（实际调用 srun_main.sh）

    Args:
        root (str): 数据集根目录。
        dataset (str): 数据集名称，例如 'caltech-101'。
        users (int): 客户端数量。
        factorization (str): 使用的矩阵分解方法名称。
        rank (int): 低秩矩阵分解的秩。
        noise (float): 差分隐私噪声级别（高斯噪声标准差）。
        seed (int): 随机种子。
        round (int): 通信轮数。
        gpus (str | None): 指定可见 GPU，如 '0' 或 '0,1'。若为 None 则不显式设置。
        exp_name (str | None): 实验名，用作 wandb group。
        task_id (str | None): 任务 ID，用于日志标识。
    """
    dataset_yaml = f'configs/datasets/{dataset}.yaml'
    prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
    wandb_group = shlex.quote(str(exp_name)) if exp_name else ""
    task_id_value = shlex.quote(str(task_id)) if task_id else '""'
    os.system(
        f'{prefix}bash srun_main.sh '
        f'{root} {dataset_yaml} {users} {factorization} {rank} {noise} {seed} {round} {wandb_group} {task_id_value}'
    )


# ==================== 实验调度函数 ====================
def run_experiment(config, gpus=None):
    """
    根据给定实验配置字典批量生成并运行所有任务组合。

    配置字典中需要包含的常用键：
        - exp_name: 实验名称（用于 wandb group）；
        - seed_list: 随机种子列表；
        - dataset_list: 数据集名称列表；
        - factorization_list: 分解方法列表；
        - noise_list: DP 噪声列表；
        - round: 通信轮数；
        - num_users 或 num_users_list: 客户端数量；
        - rank 或 rank_list: 矩阵分解秩。

    Args:
        config (dict): 实验配置字典。
        gpus (str | None): 可见 GPU 列表字符串，例如 '0' 或 '0,1'。
    """
    if config is None:
        config = EXPERIMENT_CONFIG

    # 解析 GPU 列表字符串为列表，便于轮询分配
    gpu_list = None
    if gpus:
        gpu_list = [x.strip() for x in str(gpus).split(',') if x.strip() != '']
        if len(gpu_list) == 0:
            gpu_list = None

    # 从配置中读取通用参数
    exp_name = config.get('exp_name')
    round_num = config.get('round', 20)
    users_list = config.get('num_users_list', [config.get('num_users', 10)])
    ranks = config.get('rank_list', [config.get('rank', 8)])

    # 事先统计总任务数，便于进度展示
    total_tasks = 0
    for seed in config['seed_list']:
        for dataset in config['dataset_list']:
            for users in users_list:
                for noise in config['noise_list']:
                    for factorization in config['factorization_list']:
                        for rank in ranks:
                            total_tasks += 1

    print(f"📊 实验配置: {exp_name}")
    print(f"📊 总任务数: {total_tasks}")
    print(f"🚀 开始执行实验...\n")

    # 遍历所有组合并逐个运行
    task_idx = 0
    for seed in config['seed_list']:
        for dataset in config['dataset_list']:
            for users in users_list:
                for noise in config['noise_list']:
                    for factorization in config['factorization_list']:
                        for rank in ranks:
                            task_idx += 1
                            # 轮询分配 GPU：第 i 个任务使用 gpu_list[(i-1) % len(gpu_list)]
                            gpu = None
                            if gpu_list:
                                gpu = gpu_list[(task_idx - 1) % len(gpu_list)]

                            print(
                                f"[{task_idx}/{total_tasks}] 执行任务: "
                                f"{dataset} | {factorization} | rank={rank} | "
                                f"noise={noise} | users={users} | seed={seed}"
                            )
                            if gpu:
                                print(f"   GPU: {gpu}")

                            task_id = f"[{task_idx}/{total_tasks}]"
                            run(
                                root, dataset, users, factorization, rank, noise, seed,
                                round=round_num, gpus=gpu, exp_name=exp_name, task_id=task_id
                            )

    print(f"\n✅ 实验执行完成！共完成 {total_tasks} 个任务")


# ==================== 数据集与日志工具函数 ====================
def download_datasets(base_root, dataset_name):
    """
    下载预定义标准数据集。

    Args:
        base_root (str): 数据集下载根目录。
        dataset_name (str | list[str] | None): 单个数据集名、列表或 None。
            - None: 交由下游逻辑决定；
            - str: 单个数据集；
            - list[str]: 多个数据集。
    """
    if dataset_name is None:
        dataset_list = None
    elif isinstance(dataset_name, list):
        dataset_list = dataset_name
    else:
        dataset_list = [dataset_name]
    download_standard_datasets(base_root, dataset_list)


def clean_old_logs(log_dir='logs', dry_run=False):
    """
    清理陈旧的日志文件，只保留相同参数组合下最新的一份日志。

    日志目录结构默认约定为：
        logs/{dataset}/{factorization}/{rank}_{noise}_{seed}_{num_users}_{timestamp}.log

    其中：
        - dataset      : 数据集名称（目录名）；
        - factorization: 分解方法名称（目录名）；
        - rank         : 矩阵分解秩；
        - noise        : DP 噪声级别；
        - seed         : 随机种子；
        - num_users    : 客户端数量；
        - timestamp    : 时间戳，格式为 YYYYMMDD_HHMMSS。

    同一组 (dataset, factorization, rank, noise, seed, num_users) 下，
    只保留时间戳最新的日志文件，其余全部删除。

    Args:
        log_dir (str): 日志根目录，默认为 'logs'。
        dry_run (bool): 若为 True，则仅打印将要删除的文件，不实际删除。

    Returns:
        dict: 包含统计信息的字典：
            {
                'deleted': 删除的文件数量,
                'kept':    保留的文件数量,
                'errors':  删除失败的数量
            }
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return {'deleted': 0, 'kept': 0, 'errors': 0}

    # 日志文件命名格式：{rank}_{noise}_{seed}_{num_users}_{timestamp}.log
    # 路径格式：logs/{dataset}/{factorization}/{filename}.log
    log_pattern = re.compile(r'^(\d+)_([\d.]+)_(\d+)_(\d+)_(\d{8}_\d{6})\.log$')

    # 以 (dataset, factorization, rank, noise, seed, num_users) 为分组键
    log_groups = defaultdict(list)

    # 遍历所有日志文件并分组
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
                    group_key = (dataset_name, factorization_name, rank, noise, seed, num_users)
                    log_groups[group_key].append((log_file, timestamp))
                else:
                    # 若文件名格式不符合约定，给出提示但不强制删除
                    print(f"⚠️  无法解析日志文件名格式: {log_file}")

    deleted_count = 0
    kept_count = 0
    error_count = 0

    # 对每组日志进行清理：按时间戳排序，仅保留最新一份
    for group_key, log_files in log_groups.items():
        dataset, factorization, rank, noise, seed, num_users = group_key

        if len(log_files) <= 1:
            # 只有 0 或 1 个文件时，无需清理
            kept_count += len(log_files)
            continue

        # 按时间戳降序排序：最新的在前
        log_files.sort(key=lambda x: x[1], reverse=True)

        latest_file = log_files[0][0]
        kept_count += 1  # 最新文件被保留

        # 删除该组中其他旧日志文件
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

    # 输出统计信息
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


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SepFPL experiments")

    # -------------------- 数据集相关 --------------------
    parser.add_argument(
        "-d", "--download",
        action="store_true",
        help="下载 Caltech-101、OxfordPets、OxfordFlowers、Food-101、CIFAR-100 到 root 目录"
    )

    # -------------------- 实验执行模式 --------------------
    parser.add_argument(
        "--exp1",
        action="store_true",
        help="执行实验1（包含 Simple 和 Hard 两个子实验）"
    )
    parser.add_argument(
        "--exp2",
        action="store_true",
        help="执行实验2（包含 Rank 与 HCSE/时间自适应消融两个子实验）"
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="测试单个任务（需配合 --dataset / --users / --factorization / --rank / --noise / --seed）"
    )

    # -------------------- 单任务测试参数 --------------------
    parser.add_argument("--dataset", type=str, help="数据集名称，例如 caltech-101")
    parser.add_argument("--users", type=int, help="客户端数量，例如 10")
    parser.add_argument("--factorization", type=str, help="分解方法名称，例如 sepfpl")
    parser.add_argument("--rank", type=int, help="矩阵分解的秩，例如 8")
    parser.add_argument("--noise", type=float, help="差分隐私噪声级别，例如 0.1")
    parser.add_argument("--seed", type=int, help="随机种子，例如 1")
    parser.add_argument(
        "--round", type=int, default=10,
        help="训练轮次（仅用于测试模式，默认 10）"
    )

    # -------------------- 系统/资源配置 --------------------
    parser.add_argument(
        "--gpus", type=str, default='0,1',
        help="指定可见显卡，如 '0' 或 '0,1'，多卡将用于轮询分配任务"
    )

    # -------------------- 日志管理 --------------------
    parser.add_argument(
        "--clean-logs",
        action="store_true",
        help="清理陈旧日志文件，仅保留相同参数组合下最新的一份日志"
    )
    parser.add_argument(
        "--log-dir", type=str, default='logs',
        help="日志文件根目录（与 --clean-logs 搭配使用）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要删除的日志文件，不实际删除（与 --clean-logs 搭配使用）"
    )

    # wandb 配置说明：
    # 当前脚本不再通过命令行参数配置 wandb，而是完全依赖环境变量：
    #   - WANDB_MODE, WANDB_PROJECT, WANDB_ENTITY, WANDB_GROUP, WANDB_TAGS, WANDB_DIR 等；
    #   - 设置 WANDB_DISABLED=1 可以完全禁用 wandb。

    args = parser.parse_args()

    if args.clean_logs:
        # 清理陈旧日志文件
        print("🧹 开始清理陈旧的日志文件...")
        if args.dry_run:
            print("🔍 [DRY RUN 模式] 仅预览将被删除的文件，不会实际删除\n")
        stats = clean_old_logs(log_dir=args.log_dir, dry_run=args.dry_run)
        if args.dry_run:
            print("\n💡 提示: 去掉 --dry-run 即可实际执行删除操作")
        print("\n✅ 日志清理完成！")

    elif args.download:
        # 下载标准数据集
        download_datasets(root, ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101', 'cifar-100'])

    elif args.exp1:
        # 执行实验 1（主实验）
        print("🚀 开始执行实验1...")
        print("=" * 80)
        print("实验1.1: Simple - 标准数据集 + 10 客户端")
        print("=" * 80)
        run_experiment(EXPERIMENT_1_SIMPLE_CONFIG, gpus=args.gpus)

        print("\n" + "=" * 80)
        print("实验1.2: Hard - CIFAR-100，不同客户端数量")
        print("=" * 80)
        run_experiment(EXPERIMENT_1_HARD_CONFIG, gpus=args.gpus)
        print("\n✅ 实验1执行完成！")

    elif args.exp2:
        # 执行实验 2（消融实验）
        print("🚀 开始执行实验2...")
        print("=" * 80)
        print("实验2.1: Rank 消融实验（仅 SepFPL）")
        print("=" * 80)
        run_experiment(EXPERIMENT_2_RANK_CONFIG, gpus=args.gpus)

        print("\n" + "=" * 80)
        print("实验2.2: HCSE 与时间自适应隐私分配机制消融实验")
        print("=" * 80)
        run_experiment(EXPERIMENT_2_ABLATION_CONFIG, gpus=args.gpus)
        print("\n✅ 实验2执行完成！")

    elif args.test:
        # 单任务测试模式
        required_params = ['dataset', 'users', 'factorization', 'rank', 'noise', 'seed']
        missing_params = [p for p in required_params if getattr(args, p) is None]
        if missing_params:
            print(f"❌ 错误：测试任务缺少以下参数: {', '.join(missing_params)}")
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

            print("   Wandb: 通过环境变量自动配置\n")

            # 若指定了多张卡，测试模式仅使用列表中的第一张
            gpu_for_test = None
            if args.gpus:
                gpu_list = [x.strip() for x in str(args.gpus).split(',') if x.strip() != '']
                if len(gpu_list) > 0:
                    gpu_for_test = gpu_list[0]

            run(
                root, args.dataset, args.users, args.factorization,
                args.rank, args.noise, args.seed, args.round, gpu_for_test,
                exp_name=None, task_id=None  # 测试任务不使用 exp_name 和 task_id
            )
            print("\n✅ 任务执行完成！")

    else:
        # 无任何子命令时，输出可用选项概览
        print("未指定操作。可用选项如下：")
        print("  --download    : 下载标准数据集")
        print("  --exp1        : 执行实验1（Simple + Hard）")
        print("  --exp2        : 执行实验2（Rank + Ablation）")
        print("  --test        : 测试单个任务（需配合参数）")
        print("  --clean-logs  : 清理陈旧日志，仅保留最新日志")
        print("  --dry-run     : 与 --clean-logs 搭配，仅预览待删文件")

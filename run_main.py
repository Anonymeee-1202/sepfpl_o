import os
from datasets import download_standard_datasets

# ==================== 配置参数 ====================
root = '/home/liuxin25/dataset'  # 数据集路径
users = 10  # 客户端数量

# 实验配置 - 用于测试个性化和泛化能力
EXPERIMENT_CONFIG = {
    'seed_list': [1],
    'dataset_list': ['caltech-101', 'oxford_pets', 'oxford_flowers'], # 'food-101'
    'factorization_list': ['promptfl', 'fedotp', 'fedpgp', 'dpfpl', 'sepfpl'],  # 测试的方法
    'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],  # 差分隐私噪声级别
    'rank': 8,  # 矩阵分解的秩
    'num_terminals': 2,  # 并行终端数量
    'partition_list': ['noniid-labeldir'],
    'round': 30,  # 通信轮数
}


# ==================== 核心功能函数 ====================
def run(root, dataset, users, factorization, rank, noise, seed, partition='noniid-labeldir', round=10, gpus=None):
    """运行单个实验任务"""
    dataset_yaml = f'configs/datasets/{dataset}.yaml'
    prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
    gpu_arg = f" {gpus}" if gpus else ""
    os.system(f'{prefix}bash srun_main.sh {root} {dataset_yaml} {users} {factorization} {rank} {noise} {seed} {partition} {round}{gpu_arg}')


def generate_task_commands(config):
    """生成所有任务的命令列表（不带GPU信息，GPU在terminal级别分配）"""
    tasks = []
    round_num = config.get('round', 20)  # 默认10轮
    partition_list = config.get('partition_list') or [config.get('partition', 'noniid-labeldir')]
    for seed in config['seed_list']:
        for dataset in config['dataset_list']:
            for noise in config['noise_list']:
                for factorization in config['factorization_list']:
                    for partition in partition_list:
                        task_cmd = (
                            f'bash srun_main.sh {root} configs/datasets/{dataset}.yaml {users} '
                            f'{factorization} {config["rank"]} {noise} {seed} {partition} {round_num}'
                        )
                        tasks.append(task_cmd)
    return tasks


def save_task_files(tasks, config, gpus=None):
    """将任务保存到文件，按终端分配；每个terminal分配到一张GPU"""
    # 解析GPU列表
    gpu_list = None
    if gpus:
        gpu_list = [x.strip() for x in str(gpus).split(',') if x.strip() != '']
        if len(gpu_list) == 0:
            gpu_list = None
    
    os.makedirs('tasks', exist_ok=True)
    # 清理已有的任务脚本
    removed = 0
    for fname in os.listdir('tasks'):
        if fname.endswith('.sh'):
            try:
                os.remove(os.path.join('tasks', fname))
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"🧹 Removed {removed} old task files in ./tasks/")
    
    # 保存完整任务列表
    task_file = 'tasks/task_list.sh'
    with open(task_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'# Total tasks: {len(tasks)}\n\n')
        for i, task in enumerate(tasks, 1):
            f.write(f'# Task {i}/{len(tasks)}\n')
            f.write(f'{task}\n\n')
    os.chmod(task_file, 0o755)
    
    # 分配到不同终端的任务文件（使用轮询方式）
    num_terminals = config['num_terminals']
    
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
        terminal_file = f'tasks/terminal_{terminal_id}.sh'
        
        with open(terminal_file, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Terminal {terminal_id + 1} tasks')
            if assigned_gpu is not None:
                f.write(f' (GPU {assigned_gpu})')
            f.write(f' - Total: {len(terminal_tasks[terminal_id])} tasks\n\n')
            
            for task_idx, task in terminal_tasks[terminal_id]:
                f.write(f'# Task {task_idx}/{len(tasks)}\n')
                # 为任务添加GPU前缀和参数
                if assigned_gpu is not None:
                    prefix = f"CUDA_VISIBLE_DEVICES={assigned_gpu} "
                    gpu_arg = f" {assigned_gpu}"
                    # 为任务命令添加GPU信息
                    task_with_gpu = f'{prefix}{task}{gpu_arg}'
                    f.write(f'{task_with_gpu}\n\n')
                else:
                    f.write(f'{task}\n\n')
        
        os.chmod(terminal_file, 0o755)
        gpu_info = f" (GPU {assigned_gpu})" if assigned_gpu is not None else ""
        task_indices = [idx for idx, _ in terminal_tasks[terminal_id]]
        print(f"✅ Created {terminal_file} with {len(terminal_tasks[terminal_id])} tasks {gpu_info}")
        print(f"   Task indices: {task_indices[:5]}{'...' if len(task_indices) > 5 else ''}")


# ==================== 实验相关函数 ====================
def test_generalization_and_personalization(gpus=None):
    """顺序执行个性化和泛化性测试"""
    tasks = generate_task_commands(EXPERIMENT_CONFIG)
    # 如果有GPU，为所有任务添加GPU信息
    if gpus:
        gpu_list = [x.strip() for x in str(gpus).split(',') if x.strip() != '']
        if len(gpu_list) == 1:
            prefix = f"CUDA_VISIBLE_DEVICES={gpu_list[0]} "
            gpu_arg = f" {gpu_list[0]}"
            tasks = [f'{prefix}{task}{gpu_arg}' if not task.startswith('CUDA_VISIBLE_DEVICES') else task for task in tasks]
    for task in tasks:
        os.system(task)


def generate_task_list(gpus=None):
    """生成任务列表文件，用于多终端并行执行"""
    tasks = generate_task_commands(EXPERIMENT_CONFIG)
    save_task_files(tasks, EXPERIMENT_CONFIG, gpus=gpus)
    
    print(f"\n📊 Total tasks: {len(tasks)}")
    print(f"📁 Task files created in ./tasks/")
    print(f"🚀 To run all tasks in one terminal: bash tasks/task_list.sh")
    print(f"🚀 To run in parallel terminals:")
    for terminal_id in range(EXPERIMENT_CONFIG['num_terminals']):
        print(f"   Terminal {terminal_id + 1}: bash tasks/terminal_{terminal_id}.sh")


def download_datasets(base_root, dataset_name):
    # 支持传入单个字符串或列表
    if dataset_name is None:
        dataset_list = None
    elif isinstance(dataset_name, list):
        dataset_list = dataset_name
    else:
        dataset_list = [dataset_name]
    download_standard_datasets(base_root, dataset_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DP-FPL experiments")
    parser.add_argument("-t", "--test_generalization_and_personalization", action="store_true", help="运行个性化与泛化性测试批处理")
    parser.add_argument("-s", "--single-test", action="store_true", help="运行单个测试")
    parser.add_argument("-d", "--download", action="store_true", help="下载 Caltech101、OxfordPets、OxfordFlowers 到 root 目录")
    parser.add_argument("-g", "--generate-tasks", action="store_true", help="生成任务列表文件，用于多终端并行执行")
    parser.add_argument("--gpus", type=str, default='0,1', help="指定可见显卡，如 '0' 或 '0,1'")
    parser.add_argument("--partition", type=str, default=None, help="指定数据划分策略，如 'homo'、'noniid-labeldir'")
    args = parser.parse_args()

    if args.partition:
        EXPERIMENT_CONFIG['partition_list'] = [args.partition]

    default_partition = EXPERIMENT_CONFIG['partition_list'][0]

    if args.download:
        # download_datasets(root, EXPERIMENT_CONFIG['dataset_list'])
        download_datasets(root, ['oxford_flowers'])
    elif args.generate_tasks:
        generate_task_list(gpus=args.gpus)
    elif args.test_generalization_and_personalization:
        test_generalization_and_personalization(gpus=args.gpus)
    elif args.single_test:
        for factorization in ['dpfpl']: #'fedpgp', 'promptfl'
            run(root, 'oxford_flowers', users, factorization, 8, 0.0, 1, round=3, partition=default_partition, gpus=0)
        # 'dataset_list': ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101']
        # 'factorization_list': ['sepfpl', 'dpfpl', 'fedpgp', 'promptfl', 'fedotp'] # 测试的方法
    else:
        print("未指定操作。")
        print("可用选项:")
        print("  --download: 下载数据集")
        print("  --generate-tasks: 生成任务列表文件") 
        print("  --test_generalization_and_personalization: 运行测试批处理")
        print("  --single-test: 运行单个测试")

import os
import shlex
import argparse
import itertools
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

# ==============================================================================
# SECTION 1: 全局配置 & 常量 (Configuration & Constants)
# ==============================================================================

ROOT_DIR = os.path.expanduser('~/dataset')

# 实验参数配置字典
EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'EXPERIMENT_1_STANDARD': {
        'exp_name': 'exp1-standard',
        'seed_list': [1],
        'dataset_list': ['caltech-101', 'oxford_flowers', 'food-101', 'stanford_dogs'],
        'factorization_list': ['promptfl', 'fedotp', 'fedpgp', 'dpfpl', 'sepfpl'],
        'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
        'rank_list': [8],
        'num_users_list': [10],
        'round': 40,
        'sepfpl_topk': 8,
        'rdp_p': 0.2,
    },
    'EXPERIMENT_1_EXTENSION': {
        'exp_name': 'exp1-extension',
        'seed_list': [1],
        'dataset_list': ['cifar-100'],
        'factorization_list': ['promptfl', 'fedotp', 'fedpgp', 'dpfpl', 'sepfpl'],
        'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
        'rank_list': [8],
        'num_users_list': [25, 50],
        'round': 40,
        'sepfpl_topk': 8,
        'rdp_p': 0.2,
    },
    'EXPERIMENT_2_ABLATION': {
        'exp_name': 'exp2-ablation',
        'seed_list': [1],
        'dataset_list': ['caltech-101', 'stanford_dogs', 'oxford_flowers', 'food-101'],
        'factorization_list': ['dpfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse', 'sepfpl'],
        'noise_list': [0.4, 0.1, 0.01],
        'rank_list': [8],
        'num_users_list': [10],
        'round': 40,
        'sepfpl_topk': 8,
        'rdp_p': 0.2,
    },
    # 实验3系列：敏感性分析
    'EXPERIMENT_3_RANK': {
        'exp_name': 'exp3-sens-rank',
        'seed_list': [1],
        'dataset_list': ['stanford_dogs', 'oxford_flowers'],
        'factorization_list': ['sepfpl'],
        'noise_list': [0, 0.4, 0.1, 0.01],
        'rank_list': [1, 2, 4, 8, 16],
        'num_users_list': [10],
        'round': 20,
        'sepfpl_topk': 8,
        'rdp_p': 0.2,
    },
    'EXPERIMENT_3_TOPK': {
        'exp_name': 'exp3-sens-topk',
        'seed_list': [1],
        'dataset_list': ['stanford_dogs', 'oxford_flowers'],
        'factorization_list': ['sepfpl'],
        'noise_list': [0, 0.4, 0.1, 0.01],
        'rank_list': [8],
        'num_users_list': [10],
        'round': 20,
        'sepfpl_topk_list': [2, 4, 6, 8], # 特殊列表参数
        'rdp_p': 0.2,
    },
    'EXPERIMENT_3_RDP_P': {
        'exp_name': 'exp3-sens-rdpp',
        'seed_list': [1],
        'dataset_list': ['stanford_dogs', 'oxford_flowers'],
        'factorization_list': ['sepfpl'],
        'noise_list': [0.4, 0.1, 0.01],
        'rank_list': [8],
        'num_users_list': [10],
        'round': 20,
        'sepfpl_topk': 8,
        'rdp_p_list': [0, 0.2, 0.5, 1], # 特殊列表参数
    },
    'EXPERIMENT_4_MIA': {
        'exp_name': 'exp4-mia',
        'seed_list': list(range(1, 11)),
        'dataset_list': ['caltech-101', 'stanford_dogs', 'oxford_flowers', 'food-101'],
        'factorization_list': ['sepfpl'],
        'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
        'rank_list': [8],
        'num_users_list': [10],
        'round': 10,
        'sepfpl_topk': 8,
        'rdp_p': 0.2,
    },
}

EXP_ARG_MAP = {
    'exp1': (['EXPERIMENT_1_STANDARD', 'EXPERIMENT_1_EXTENSION'], "实验1 (Standard + Extension)"),
    'exp2': (['EXPERIMENT_2_ABLATION'], "实验2 (机制消融)"),
    'exp3': (['EXPERIMENT_3_RANK', 'EXPERIMENT_3_TOPK', 'EXPERIMENT_3_RDP_P'], "实验3 (敏感性分析 - 全部合并)"),
    'exp3_rank': (['EXPERIMENT_3_RANK'], "实验3.1 (Rank)"),
    'exp3_topk': (['EXPERIMENT_3_TOPK'], "实验3.2 (TopK)"),
    'exp3_rdp_p': (['EXPERIMENT_3_RDP_P'], "实验3.3 (RDP P)"),
    'exp4': (['EXPERIMENT_4_MIA'], "实验4 (MIA)"),
}

# ==============================================================================
# SECTION 2: 数据结构 & 基础类 (Data Structures & Base Classes)
# ==============================================================================

@dataclass
class TaskStep:
    """定义任务中的一个步骤（例如：MIA中的 target, shadow, attack）"""
    name: str
    command: str

@dataclass
class Task:
    """定义一个完整的执行任务（可能包含多个串行步骤）"""
    task_id: str
    description: str
    steps: List[TaskStep]
    gpu: Optional[str] = None
    
    # 用于去重的唯一标识符 (基于关键参数)
    unique_key: str = "" 

class CommandBuilder:
    """构建 Shell 命令的工具类"""
    
    @staticmethod
    def build(
        script_name: str,
        dataset: str, users: int, factorization: str, rank: int,
        noise: float, seed: int, round_num: int, 
        exp_name: str = "", task_id: str = "",
        extra_args: List[str] = None,
        env_vars: Dict[str, str] = None
    ) -> str:
        dataset_yaml = f'configs/datasets/{dataset}.yaml'
        
        parts = ["bash", script_name, shlex.quote(ROOT_DIR), shlex.quote(dataset_yaml)]
        parts.extend([str(users), shlex.quote(factorization), str(rank)])
        parts.extend([str(noise), str(seed), str(round_num)])
        parts.extend([shlex.quote(exp_name), shlex.quote(task_id)])
        
        if extra_args:
            parts.extend([str(arg) for arg in extra_args])
            
        cmd_str = " ".join(parts)
        
        if env_vars:
            env_str = " ".join([f"{k}={v}" for k, v in env_vars.items()])
            return f"{env_str} {cmd_str}"
        return cmd_str

# ==============================================================================
# SECTION 3: 任务生成逻辑 (Task Generation Logic)
# ==============================================================================

def generate_tasks_for_config(
    config_key: str, 
    config: Dict[str, Any], 
    gpu_pool: List[str],
    mia_flags: Dict[str, bool] = None
) -> List[Task]:
    """根据配置字典生成任务列表。统一处理 Standard 和 MIA 任务。"""
    
    tasks: List[Task] = []
    
    # 提取基础参数列表
    seed_list = config.get('seed_list', [1])
    dataset_list = config.get('dataset_list', [])
    factorization_list = config.get('factorization_list', [])
    noise_list = config.get('noise_list', [0.0])
    users_list = config.get('num_users_list') or [10]
    rank_list = config.get('rank_list') or [8]
    
    # 提取特殊参数 (单一值或列表)
    sepfpl_topk_list = config.get('sepfpl_topk_list') or ([config.get('sepfpl_topk')] if config.get('sepfpl_topk') is not None else [None])
    rdp_p_list = config.get('rdp_p_list') or ([config.get('rdp_p')] if config.get('rdp_p') is not None else [None])
    
    # 生成笛卡尔积
    combinations = list(itertools.product(
        seed_list, dataset_list, users_list, rank_list, noise_list, factorization_list,
        sepfpl_topk_list, rdp_p_list
    ))
    
    total_combs = len(combinations)
    is_mia = 'MIA' in config_key
    
    # --- MIA 逻辑 ---
    if is_mia:
        # MIA 任务分为两类：Target/Shadow (依赖 Seed) 和 Attack (跨 Seed, 通常只跑一次)
        # 1. Target & Shadow Tasks
        for idx, (seed, ds, u, r, n, fact, topk, rdpp) in enumerate(combinations, 1):
            if not (mia_flags['fed_train'] or mia_flags['generate_shadow']): continue
            
            steps = []
            common_args = [topk if topk is not None else '""', rdpp if rdpp is not None else '""']
            base_mia_args = {
                'script_name': 'srun_mia.sh', 'dataset': ds, 'users': u, 'factorization': fact,
                'rank': r, 'noise': n, 'seed': seed, 'round_num': config.get('round', 10),
                'exp_name': config['exp_name']
            }

            if mia_flags['fed_train']:
                cmd = CommandBuilder.build(**base_mia_args, task_id='target', extra_args=['target', '--skip-test'] + common_args)
                steps.append(TaskStep('Train Target', cmd))
            
            if mia_flags['generate_shadow']:
                # Shadow 模式下脚本需要 'generate_shadow' 参数
                cmd = CommandBuilder.build(**base_mia_args, task_id='shadow', extra_args=['generate_shadow', '""'] + [f"--sepfpl-topk {topk}" if topk else "", f"--rdp-p {rdpp}" if rdpp else ""])
                steps.append(TaskStep('Gen Shadow', cmd))

            if steps:
                gpu = gpu_pool[(idx - 1) % len(gpu_pool)] if gpu_pool else None
                task_id = f"[{idx}/{total_combs}]"
                desc = f"MIA-Fed | {ds} | {fact} | s={seed}"
                # Unique Key 用于去重，包含所有参数
                ukey = f"{ds}-{u}-{fact}-{r}-{n}-{seed}-{topk}-{rdpp}-fed"
                tasks.append(Task(task_id, desc, steps, gpu, ukey))

        # 2. Attack Tasks (去重 Seed)
        if mia_flags['attack_train'] or mia_flags['attack_test']:
            seen_attacks = set()
            attack_tasks_temp = []
            
            for (seed, ds, u, r, n, fact, topk, rdpp) in combinations:
                key = (ds, u, r, n, fact, topk, rdpp)
                if key in seen_attacks: continue
                seen_attacks.add(key)
                
                # Attack 通常只用第一个 seed
                first_seed = seed 
                steps = []
                base_mia_args = {
                    'script_name': 'srun_mia.sh', 'dataset': ds, 'users': u, 'factorization': fact,
                    'rank': r, 'noise': n, 'seed': first_seed, 'round_num': config.get('round', 10),
                    'exp_name': config['exp_name']
                }
                extra_flags = [f"--sepfpl-topk {topk}" if topk else "", f"--rdp-p {rdpp}" if rdpp else ""]

                if mia_flags['attack_train']:
                    cmd = CommandBuilder.build(**base_mia_args, task_id='attack', extra_args=['train', '""'] + extra_flags)
                    steps.append(TaskStep('Train Attack', cmd))
                
                if mia_flags['attack_test']:
                    cmd = CommandBuilder.build(**base_mia_args, task_id='attack', extra_args=['test', '""'] + extra_flags)
                    steps.append(TaskStep('Test Attack', cmd))

                if steps:
                    desc = f"MIA-Attack | {ds} | {fact} | n={n}"
                    ukey = f"{ds}-{u}-{fact}-{r}-{n}-attack-{topk}-{rdpp}"
                    attack_tasks_temp.append(Task("", desc, steps, None, ukey)) # ID 和 GPU 稍后分配
            
            # 分配 GPU 给 Attack 任务 (Attack 任务通常较快，或需要并行)
            for i, t in enumerate(attack_tasks_temp):
                t.task_id = f"[Attack-{i+1}/{len(attack_tasks_temp)}]"
                t.gpu = gpu_pool[i % len(gpu_pool)] if gpu_pool else None
                tasks.append(t)

    # --- 标准实验逻辑 ---
    else:
        for idx, (seed, ds, u, r, n, fact, topk, rdpp) in enumerate(combinations, 1):
            gpu = gpu_pool[(idx - 1) % len(gpu_pool)] if gpu_pool else None
            
            extra = []
            if topk is not None: extra.append(str(topk))
            else: extra.append('""')
            if rdpp is not None: extra.append(str(rdpp))
            else: extra.append('""')

            cmd = CommandBuilder.build(
                'srun_main.sh', ds, u, fact, r, n, seed, config.get('round', 40),
                config['exp_name'], f"[{idx}/{total_combs}]", extra_args=extra
            )
            
            desc = f"{ds} | {fact} | n={n} | s={seed}"
            if topk is not None: desc += f" | topk={topk}"
            if rdpp is not None: desc += f" | p={rdpp}"
            
            ukey = f"{ds}-{u}-{fact}-{r}-{n}-{seed}-{topk}-{rdpp}"
            tasks.append(Task(f"[{idx}/{total_combs}]", desc, [TaskStep("Main", cmd)], gpu, ukey))
            
    return tasks

# ==============================================================================
# SECTION 4: 脚本生成器 (Script Writer)
# ==============================================================================

class ScriptWriter:
    @staticmethod
    def write(tasks: List[Task], output_path: str, exp_title: str, num_parallel_threads: int = None):
        """
        生成 Bash 脚本。
        支持：串行模式、GPU并行模式、以及多线程并行模式 (Thread grouping)。
        """
        # 按 GPU 分组
        tasks_by_gpu = defaultdict(list)
        for t in tasks:
            key = t.gpu if t.gpu else "cpu"
            tasks_by_gpu[key].append(t)
            
        # 如果启用线程并行，对每个 GPU 的任务进行分组
        if num_parallel_threads and num_parallel_threads > 1:
            strategy_desc = f"并行线程模式 (每GPU {num_parallel_threads} 任务)"
            # 将 List[Task] 转换为 List[List[Task]] (Chunking)
            grouped_tasks_by_gpu = {}
            for gpu, task_list in tasks_by_gpu.items():
                chunks = [task_list[i:i + num_parallel_threads] for i in range(0, len(task_list), num_parallel_threads)]
                grouped_tasks_by_gpu[gpu] = chunks
        else:
            strategy_desc = "GPU 并行模式 (同一GPU串行)"
            grouped_tasks_by_gpu = {g: [[t] for t in tl] for g, tl in tasks_by_gpu.items()} # 每个 chunk 只有一个任务

        with open(output_path, 'w', encoding='utf-8') as f:
            # --- Header ---
            f.write(f"""#!/bin/bash
# 实验: {exp_title}
# 生成时间: {datetime.now()}
# 任务总数: {len(tasks)}
# 策略: {strategy_desc}
# ----------------------------------------------------

""")

            # --- Workers Definition ---
            sorted_gpus = sorted(grouped_tasks_by_gpu.keys(), key=lambda x: (len(x), x))
            
            for gpu_key in sorted_gpus:
                worker_name = f"run_worker_{gpu_key}".replace(',', '_') # handle multi-gpu string
                chunks = grouped_tasks_by_gpu[gpu_key]
                
                f.write(f"{worker_name}() {{\n")
                f.write(f"    echo '🚀 [Worker {gpu_key}] 启动，共 {len(chunks)} 组任务'\n")
                
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"    # --- Group {i}/{len(chunks)} ---\n")
                    f.write("    pids=()\n")
                    
                    for task in chunk:
                        env_prefix = f"CUDA_VISIBLE_DEVICES={task.gpu} " if task.gpu and task.gpu != 'cpu' else ""
                        f.write(f"    # Task: {task.description}\n")
                        
                        # 如果任务有多个步骤，需要用 () 组合成子shell或 && 连接
                        # 这里使用 simple && execution chain inside a background block
                        cmds_chain = " && ".join([f"echo '  -> {s.name}' && {s.command}" for s in task.steps])
                        
                        # 后台执行整个任务链
                        f.write(f"    ({env_prefix}{cmds_chain}) &\n")
                        f.write(f"    pids+=($!)\n")
                    
                    # 等待该组所有任务完成 (Sync point)
                    f.write(f"\n    echo '⏳ [Worker {gpu_key}] 等待第 {i} 组任务完成...'\n")
                    f.write("    for pid in \"${pids[@]}\"; do wait \"$pid\"; done\n")
                    f.write(f"    echo '✅ [Worker {gpu_key}] 第 {i} 组完成'\n\n")
                
                f.write(f"    echo '🎉 [Worker {gpu_key}] 所有任务完成'\n")
                f.write("}\n\n")

            # --- Execution ---
            f.write("echo '================ 开始执行 ================'\n")
            for gpu_key in sorted_gpus:
                worker_name = f"run_worker_{gpu_key}".replace(',', '_')
                f.write(f"{worker_name} &\n")
            
            f.write("\nwait\necho '🏁 所有 Worker 已退出。'\n")
            
        os.chmod(output_path, 0o755)
        return output_path

# ==============================================================================
# SECTION 5: 主程序 (Main Execution)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SepFPL 实验管理工具 (Refactored)")
    
    # 模式
    parser.add_argument("-d", "--download", action="store_true", help="下载数据集")
    parser.add_argument("-t", "--test", action="store_true", help="测试模式 (单任务)")
    
    # 实验选择
    grp_exp = parser.add_argument_group("实验选择")
    for flag, (_, desc) in EXP_ARG_MAP.items():
        grp_exp.add_argument(f"--{flag.replace('_', '-')}", action="store_true", help=desc)
        
    # 配置
    parser.add_argument("--gpus", type=str, default='0,1', help="可用 GPU (e.g. '0,1')")
    parser.add_argument("--threads", type=int, default=1, help="并行度: 每个GPU同时运行几个任务 (默认1=串行)")
    
    # MIA 控制
    grp_mia = parser.add_argument_group("MIA 阶段控制")
    grp_mia.add_argument("--fed-train", action="store_true")
    grp_mia.add_argument("--generate-shadow", action="store_true")
    grp_mia.add_argument("--attack-train", action="store_true")
    grp_mia.add_argument("--attack-test", action="store_true")
    
    # 测试参数
    grp_test = parser.add_argument_group("测试模式参数")
    grp_test.add_argument("--dataset", type=str)
    grp_test.add_argument("--users", type=int)
    grp_test.add_argument("--factorization", type=str)
    grp_test.add_argument("--rank", type=int)
    grp_test.add_argument("--noise", type=float)
    grp_test.add_argument("--seed", type=int)

    args = parser.parse_args()

    # --- 1. 下载模式 ---
    if args.download:
        try:
            from datasets import download_standard_datasets
            download_standard_datasets(ROOT_DIR, ['caltech-101', 'oxford_flowers', 'food-101', 'cifar-100'])
        except ImportError:
            print("❌ 缺少 datasets 库")
        return

    # --- 2. 测试模式 ---
    if args.test:
        if not all([args.dataset, args.users, args.factorization, args.rank is not None, args.noise is not None, args.seed]):
            print("❌ 测试模式缺少参数")
            return
        
        cmd = CommandBuilder.build(
            'srun_main.sh', args.dataset, args.users, args.factorization, args.rank, 
            args.noise, args.seed, 5, "test-exp", "[TEST]"
        )
        print(f"🧪 执行测试: {cmd}")
        subprocess.run(cmd, shell=True)
        return

    # --- 3. 批量生成模式 ---
    configs_to_run = []
    for flag, (keys, _) in EXP_ARG_MAP.items():
        if getattr(args, flag.replace('-', '_'), False):
            configs_to_run.extend(keys)
    
    # 去重并保持顺序
    configs_to_run = list(dict.fromkeys(configs_to_run))
    
    if not configs_to_run:
        print("⚠️ 未选择实验。使用 --exp1, --exp2 等参数。")
        return

    gpu_pool = [g.strip() for g in args.gpus.split(',')] if args.gpus else []
    mia_flags = {
        'fed_train': args.fed_train, 'generate_shadow': args.generate_shadow,
        'attack_train': args.attack_train, 'attack_test': args.attack_test
    }
    # MIA 默认行为
    if not any(mia_flags.values()):
        mia_flags['generate_shadow'] = True
        mia_flags['attack_train'] = True

    # 收集所有任务
    all_tasks: List[Task] = []
    for cfg_key in configs_to_run:
        print(f"⚙️  处理配置: {cfg_key}")
        tasks = generate_tasks_for_config(
            cfg_key, EXPERIMENT_CONFIGS[cfg_key], gpu_pool, mia_flags
        )
        all_tasks.extend(tasks)

    # 全局去重 (基于 unique_key)
    # 这自动解决了 exp3 合并的问题，只要生成的 unique_key 一致
    unique_tasks = {}
    for t in all_tasks:
        if t.unique_key not in unique_tasks:
            unique_tasks[t.unique_key] = t
        else:
            # 如果同一个任务在不同实验配置中出现（例如 exp3 合并），
            # 我们可以保留现有的，或者简单跳过。这里简单保留第一次出现的。
            pass
            
    final_task_list = list(unique_tasks.values())
    
    if not final_task_list:
        print("❌ 未生成任何任务")
        return

    # 生成脚本
    script_name = "batch_run.sh"
    if len(configs_to_run) == 1:
        script_name = f"run_{EXPERIMENT_CONFIGS[configs_to_run[0]]['exp_name']}.sh"
    elif "EXPERIMENT_3_RANK" in configs_to_run: # 简单的启发式命名
        script_name = "run_exp3_merged.sh"
        
    out_path = os.path.join("scripts", script_name)
    os.makedirs("scripts", exist_ok=True)
    
    path = ScriptWriter.write(final_task_list, out_path, str(configs_to_run), args.threads)
    print(f"\n✅ 脚本已生成: {path}")
    print(f"   任务总数: {len(final_task_list)}")
    print(f"   执行方式: bash {path}")

if __name__ == "__main__":
    main()
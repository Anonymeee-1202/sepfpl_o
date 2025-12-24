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
        'seed_list': list(range(11, 15)),
        'dataset_list': ['caltech-101', 'stanford_dogs', 'oxford_flowers', 'food-101'],
        'factorization_list': ['sepfpl'],
        'noise_list': [0.0, 0.4, 0.2, 0.1, 0.05, 0.01],
        'rank_list': [8],
        'num_users_list': [10],
        'round': 10,
        'sepfpl_topk': 8,
        'rdp_p': 0.2,
        'shadow_sample_ratio_list': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    'EXPERIMENT_5_GRADIENT_CLUSTERING': {
        'exp_name': 'exp5-gradient-clustering',
        'seed_list': [1],
        'dataset_list': ['cifar-100'], # 'stanford_dogs', 'food-101', 
        'factorization_list': ['sepfpl'],
        'noise_list': [0.0, 0.1, 0.01], 
        'rank_list': [8],
        'num_users_list': [400],
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
    'exp5': (['EXPERIMENT_5_GRADIENT_CLUSTERING'], "实验5 (梯度聚类可视化)"),
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
    
    # 是否允许在同一个 GPU 上并行执行多个任务（仅用于 attack_train）
    allow_parallel: bool = False
    
    # MIA 实验的阶段标识：'fed_train', 'generate_shadow', 'attack_train', 'attack_test'
    # 用于确保执行顺序：fed_train -> generate_shadow -> attack_train -> attack_test
    stage: Optional[str] = None 

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
        extra_args = extra_args or []
        
        # srun_mia.sh 需要 mode 作为第一个参数（在 root 之前）
        # srun_main.sh 不需要 mode
        if script_name == 'srun_mia.sh':
            # 提取 mode（应该是 extra_args 的第一个参数）
            mode = extra_args[0] if extra_args else 'train'
            remaining_extra = extra_args[1:] if len(extra_args) > 1 else []
            
            parts = ["bash", script_name, mode, shlex.quote(ROOT_DIR), shlex.quote(dataset_yaml)]
            parts.extend([str(users), shlex.quote(factorization), str(rank)])
            parts.extend([str(noise), str(seed), str(round_num)])
            parts.extend([shlex.quote(exp_name) if exp_name else '""', shlex.quote(task_id) if task_id else '""'])
            # 额外参数（如 --sepfpl-topk, --rdp-p, --noise-list, --shadow-sample-ratio-list）作为 $12+
            if remaining_extra:
                parts.extend([str(arg) for arg in remaining_extra if arg])  # 过滤空字符串
        else:
            # srun_main.sh 的标准格式
            parts = ["bash", script_name, shlex.quote(ROOT_DIR), shlex.quote(dataset_yaml)]
            parts.extend([str(users), shlex.quote(factorization), str(rank)])
            parts.extend([str(noise), str(seed), str(round_num)])
            parts.extend([shlex.quote(exp_name) if exp_name else '""', shlex.quote(task_id) if task_id else '""'])
            # sepfpl_topk 和 rdp_p 作为 $11 和 $12
            if extra_args:
                parts.extend([str(arg) for arg in extra_args if arg])  # 过滤空字符串
            
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
    
    is_mia = 'MIA' in config_key
    # 对于 MIA 实验，处理 shadow_sample_ratio_list 与 noise_list 的对应关系
    shadow_sample_ratio_list = []  # 用于attack_train阶段
    if is_mia:
        shadow_sample_ratio_list = config.get('shadow_sample_ratio_list', [])
        if shadow_sample_ratio_list:
            # 确保 noise_list 和 shadow_sample_ratio_list 长度一致
            if len(noise_list) != len(shadow_sample_ratio_list):
                raise ValueError(f"MIA实验配置错误: noise_list长度({len(noise_list)})与shadow_sample_ratio_list长度({len(shadow_sample_ratio_list)})不一致")
            # 将 noise 和 shadow_sample_ratio 配对（用于generate_shadow阶段）
            noise_shadow_pairs = list(zip(noise_list, shadow_sample_ratio_list))
        else:
            # 如果没有 shadow_sample_ratio_list，使用单个值或默认值
            shadow_sample_ratio = config.get('shadow_sample_ratio', None)
            noise_shadow_pairs = [(n, shadow_sample_ratio) for n in noise_list]
    else:
        shadow_sample_ratio = config.get('shadow_sample_ratio', None)
        noise_shadow_pairs = None
    
    # 生成笛卡尔积
    if is_mia and noise_shadow_pairs:
        # MIA 实验：使用配对的 noise 和 shadow_sample_ratio
        combinations = []
        for seed, ds, u, r, fact, topk, rdpp in itertools.product(
            seed_list, dataset_list, users_list, rank_list, factorization_list,
            sepfpl_topk_list, rdp_p_list
        ):
            for noise, shadow_ratio in noise_shadow_pairs:
                combinations.append((seed, ds, u, r, noise, fact, topk, rdpp, shadow_ratio))
    else:
        # 标准实验：使用原有的笛卡尔积
        combinations = list(itertools.product(
            seed_list, dataset_list, users_list, rank_list, noise_list, factorization_list,
            sepfpl_topk_list, rdp_p_list
        ))
        # 为兼容性，为每个组合添加 None 作为 shadow_sample_ratio
        combinations = [(*comb, None) for comb in combinations]
    
    total_combs = len(combinations)
    
    # --- MIA 逻辑 ---
    if is_mia:
        # MIA 任务分为4个阶段，确保执行顺序：
        # 1. fed_train (可选)
        # 2. generate_shadow (完整生成所有shadow数据)
        # 3. attack_train (并行执行)
        # 4. attack_test
        
        # 阶段1: fed_train (可选)
        if mia_flags['fed_train']:
            for idx, comb in enumerate(combinations, 1):
                seed, ds, u, r, n, fact, topk, rdpp, shadow_ratio = comb
                common_args = [topk if topk is not None else '""', rdpp if rdpp is not None else '""']
                # MIA 实验时跳过测试以加快训练速度
                common_args.append('--skip-test')
                cmd = CommandBuilder.build(
                    'srun_main.sh', ds, u, fact, r, n, seed, config.get('round', 10),
                    config['exp_name'], f"[{idx}/{total_combs}]", extra_args=common_args
                )
                gpu = gpu_pool[(idx - 1) % len(gpu_pool)] if gpu_pool else None
                task_id = f"[Fed-{idx}/{total_combs}]"
                desc = f"MIA-Fed | {ds} | {fact} | s={seed} | n={n}"
                ukey = f"{ds}-{u}-{fact}-{r}-{n}-{seed}-{topk}-{rdpp}-fed"
                tasks.append(Task(task_id, desc, [TaskStep('Train Target', cmd)], gpu, ukey, stage='fed_train'))
        
        # 阶段2: generate_shadow (完整生成所有shadow数据)
        if mia_flags['generate_shadow']:
            for idx, comb in enumerate(combinations, 1):
                seed, ds, u, r, n, fact, topk, rdpp, shadow_ratio = comb
                base_mia_args = {
                    'script_name': 'srun_mia.sh', 'dataset': ds, 'users': u, 'factorization': fact,
                    'rank': r, 'noise': n, 'seed': seed, 'round_num': config.get('round', 10),
                    'exp_name': config['exp_name'], 'task_id': f"[Shadow-{idx}/{total_combs}]"
                }
                extra_flags = []
                if topk is not None:
                    extra_flags.append(f"--sepfpl-topk {topk}")
                if rdpp is not None:
                    extra_flags.append(f"--rdp-p {rdpp}")
                cmd = CommandBuilder.build(**base_mia_args, extra_args=['generate_shadow'] + extra_flags)
                gpu = gpu_pool[(idx - 1) % len(gpu_pool)] if gpu_pool else None
                task_id = f"[Shadow-{idx}/{total_combs}]"
                desc = f"MIA-Shadow | {ds} | {fact} | s={seed} | n={n} | ratio={shadow_ratio}"
                ukey = f"{ds}-{u}-{fact}-{r}-{n}-{seed}-{topk}-{rdpp}-{shadow_ratio}-shadow"
                tasks.append(Task(task_id, desc, [TaskStep('Gen Shadow', cmd)], gpu, ukey, stage='generate_shadow'))
        
        # 阶段3: attack_train (每个数据集只创建一个任务，使用所有shadow_sample_ratio合成训练数据)
        if mia_flags['attack_train']:
            seen_attacks = set()
            attack_train_tasks = []
            
            # 为每个数据集创建一个训练任务
            for ds in dataset_list:
                for u in users_list:
                    for r in rank_list:
                        for fact in factorization_list:
                            for topk in sepfpl_topk_list:
                                for rdpp in rdp_p_list:
                                    # 去重key：每个数据集只创建一个任务
                                    key = (ds, u, r, fact, topk, rdpp)
                                    if key in seen_attacks: continue
                                    seen_attacks.add(key)
                                    
                                    # Attack 训练时使用第一个seed和第一个noise（仅用于参数传递）
                                    first_seed = seed_list[0] if seed_list else 1
                                    first_noise = noise_list[0] if noise_list else 0.0
                                    
                                    base_mia_args = {
                                        'script_name': 'srun_mia.sh', 'dataset': ds, 'users': u, 'factorization': fact,
                                        'rank': r, 'noise': first_noise, 'seed': first_seed, 'round_num': config.get('round', 10),
                                        'exp_name': config['exp_name']
                                    }
                                    
                                    # 传递所有的noise_list和shadow_sample_ratio_list
                                    extra_flags = []
                                    if topk is not None:
                                        extra_flags.append(f"--sepfpl-topk {topk}")
                                    if rdpp is not None:
                                        extra_flags.append(f"--rdp-p {rdpp}")
                                    
                                    # 传递noise_list和shadow_sample_ratio_list
                                    if noise_list:
                                        noise_str = ','.join(map(str, noise_list))
                                        extra_flags.append(f"--noise-list {noise_str}")
                                    if shadow_sample_ratio_list:
                                        ratio_str = ','.join(map(str, shadow_sample_ratio_list))
                                        extra_flags.append(f"--shadow-sample-ratio-list {ratio_str}")
                                    
                                    cmd = CommandBuilder.build(**base_mia_args, task_id='attack', extra_args=['train'] + extra_flags)
                                    desc = f"MIA-Attack-Train | {ds} | {fact} | (all noise, all ratios)"
                                    ukey = f"{ds}-{u}-{fact}-{r}-attack-train-{topk}-{rdpp}"
                                    attack_train_tasks.append(Task("", desc, [TaskStep('Train Attack', cmd)], None, ukey, allow_parallel=True, stage='attack_train'))
            
            # 分配 GPU 给 Attack Train 任务
            for i, t in enumerate(attack_train_tasks):
                t.task_id = f"[AttackTrain-{i+1}/{len(attack_train_tasks)}]"
                t.gpu = gpu_pool[i % len(gpu_pool)] if gpu_pool else None
                tasks.append(t)
        
        # 阶段4: attack_test
        if mia_flags['attack_test']:
            seen_attacks = set()
            attack_test_tasks = []
            
            for comb in combinations:
                seed, ds, u, r, n, fact, topk, rdpp, shadow_ratio = comb
                key = (ds, u, r, n, fact, topk, rdpp, shadow_ratio)
                if key in seen_attacks: continue
                seen_attacks.add(key)
                
                # Attack 通常只用第一个 seed
                first_seed = seed_list[0] if seed_list else 1
                base_mia_args = {
                    'script_name': 'srun_mia.sh', 'dataset': ds, 'users': u, 'factorization': fact,
                    'rank': r, 'noise': n, 'seed': first_seed, 'round_num': config.get('round', 10),
                    'exp_name': config['exp_name']
                }
                extra_flags = []
                if topk is not None:
                    extra_flags.append(f"--sepfpl-topk {topk}")
                if rdpp is not None:
                    extra_flags.append(f"--rdp-p {rdpp}")
                
                cmd = CommandBuilder.build(**base_mia_args, task_id='attack', extra_args=['test'] + extra_flags)
                desc = f"MIA-Attack-Test | {ds} | {fact} | n={n} | ratio={shadow_ratio}"
                ukey = f"{ds}-{u}-{fact}-{r}-{n}-attack-test-{topk}-{rdpp}-{shadow_ratio}"
                attack_test_tasks.append(Task("", desc, [TaskStep('Test Attack', cmd)], None, ukey, stage='attack_test'))
            
            # 分配 GPU 给 Attack Test 任务
            for i, t in enumerate(attack_test_tasks):
                t.task_id = f"[AttackTest-{i+1}/{len(attack_test_tasks)}]"
                t.gpu = gpu_pool[i % len(gpu_pool)] if gpu_pool else None
                tasks.append(t)

    # --- 标准实验逻辑 ---
    else:
        # 检查是否是梯度聚类实验，如果是则跳过测试
        is_gradient_clustering = 'GRADIENT_CLUSTERING' in config_key
        
        for idx, comb in enumerate(combinations, 1):
            seed, ds, u, r, n, fact, topk, rdpp, _ = comb
            gpu = gpu_pool[(idx - 1) % len(gpu_pool)] if gpu_pool else None
            
            extra = []
            if topk is not None: extra.append(str(topk))
            else: extra.append('""')
            if rdpp is not None: extra.append(str(rdpp))
            else: extra.append('""')
            
            # 梯度聚类实验跳过测试以加快训练速度
            if is_gradient_clustering:
                extra.append('--skip-test')

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
        # 检查是否有MIA任务（有stage字段）
        has_mia_tasks = any(t.stage is not None for t in tasks)
        
        if has_mia_tasks:
            # MIA任务：按阶段分组，确保执行顺序
            # 阶段顺序：fed_train -> generate_shadow -> attack_train -> attack_test
            stage_order = ['fed_train', 'generate_shadow', 'attack_train', 'attack_test']
            
            # 按阶段分组任务
            tasks_by_stage = defaultdict(list)
            for t in tasks:
                if t.stage:
                    tasks_by_stage[t.stage].append(t)
                else:
                    # 非MIA任务放在最后
                    tasks_by_stage['other'].append(t)
            
            # 按GPU和阶段分组
            grouped_tasks_by_gpu = {}
            for gpu_key in set(t.gpu if t.gpu else "cpu" for t in tasks):
                chunks = []
                
                # 按阶段顺序处理
                for stage in stage_order:
                    if stage not in tasks_by_stage:
                        continue
                    
                    # 获取该阶段在该GPU上的任务
                    stage_tasks = [t for t in tasks_by_stage[stage] 
                                  if (t.gpu if t.gpu else "cpu") == gpu_key]
                    
                    if not stage_tasks:
                        continue
                    
                    # attack_train 阶段允许并行
                    if stage == 'attack_train' and num_parallel_threads and num_parallel_threads > 1:
                        # 将attack_train任务分组并行执行
                        parallel_chunks = [stage_tasks[i:i + num_parallel_threads] 
                                         for i in range(0, len(stage_tasks), num_parallel_threads)]
                        chunks.extend(parallel_chunks)
                    else:
                        # 其他阶段串行执行
                        chunks.extend([[t] for t in stage_tasks])
                
                # 处理非MIA任务
                if 'other' in tasks_by_stage:
                    other_tasks = [t for t in tasks_by_stage['other'] 
                                 if (t.gpu if t.gpu else "cpu") == gpu_key]
                    chunks.extend([[t] for t in other_tasks])
                
                if chunks:
                    grouped_tasks_by_gpu[gpu_key] = chunks
        else:
            # 非MIA任务：按 GPU 分组
            tasks_by_gpu = defaultdict(list)
            for t in tasks:
                key = t.gpu if t.gpu else "cpu"
                tasks_by_gpu[key].append(t)
            
            # 根据任务的 allow_parallel 标志决定是否并行执行
            grouped_tasks_by_gpu = {}
            for gpu, task_list in tasks_by_gpu.items():
                # 将任务分为两类：允许并行的和不允许并行的
                parallel_tasks = [t for t in task_list if t.allow_parallel]
                serial_tasks = [t for t in task_list if not t.allow_parallel]
                
                chunks = []
                
                # 处理允许并行的任务
                if parallel_tasks and num_parallel_threads and num_parallel_threads > 1:
                    parallel_chunks = [parallel_tasks[i:i + num_parallel_threads] 
                                     for i in range(0, len(parallel_tasks), num_parallel_threads)]
                    chunks.extend(parallel_chunks)
                else:
                    chunks.extend([[t] for t in parallel_tasks])
                
                # 处理不允许并行的任务
                chunks.extend([[t] for t in serial_tasks])
                
                grouped_tasks_by_gpu[gpu] = chunks
        
        # 生成策略描述
        if has_mia_tasks:
            if num_parallel_threads and num_parallel_threads > 1:
                strategy_desc = f"MIA阶段模式 (fed_train -> generate_shadow -> attack_train并行({num_parallel_threads}/GPU) -> attack_test)"
            else:
                strategy_desc = "MIA阶段模式 (fed_train -> generate_shadow -> attack_train -> attack_test, 串行)"
        else:
            has_parallel = any(t.allow_parallel for t in tasks)
            if has_parallel and num_parallel_threads and num_parallel_threads > 1:
                strategy_desc = f"混合模式 (attack_train: 每GPU {num_parallel_threads} 任务并行, 其他: 串行)"
            else:
                strategy_desc = "GPU 并行模式 (同一GPU串行)"

        with open(output_path, 'w', encoding='utf-8') as f:
            # --- Header ---
            f.write(f"""#!/bin/bash
# 实验: {exp_title}
# 生成时间: {datetime.now()}
# 任务总数: {len(tasks)}
# 策略: {strategy_desc}
# ----------------------------------------------------

""")

            if has_mia_tasks:
                # MIA模式：按阶段执行，确保全局阶段同步
                stage_order = ['fed_train', 'generate_shadow', 'attack_train', 'attack_test']
                stage_names = {
                    'fed_train': '阶段1: Fed Train',
                    'generate_shadow': '阶段2: Generate Shadow',
                    'attack_train': '阶段3: Attack Train (并行)',
                    'attack_test': '阶段4: Attack Test'
                }
                
                # 按阶段和GPU组织任务
                tasks_by_stage_gpu = defaultdict(lambda: defaultdict(list))
                for t in tasks:
                    if t.stage:
                        gpu_key = t.gpu if t.gpu else "cpu"
                        tasks_by_stage_gpu[t.stage][gpu_key].append(t)
                    elif 'other' not in tasks_by_stage_gpu:
                        # 非MIA任务
                        gpu_key = t.gpu if t.gpu else "cpu"
                        tasks_by_stage_gpu['other'][gpu_key].append(t)
                
                sorted_gpus = sorted(set(t.gpu if t.gpu else "cpu" for t in tasks), key=lambda x: (len(x), x))
                
                # 为每个阶段生成执行代码
                for stage in stage_order:
                    if stage not in tasks_by_stage_gpu:
                        continue
                    
                    stage_tasks_by_gpu = tasks_by_stage_gpu[stage]
                    stage_name = stage_names.get(stage, f'阶段: {stage}')
                    
                    f.write(f"\n# ========== {stage_name} ==========\n")
                    f.write(f"echo '================ {stage_name} ================'\n")
                    
                    # 为每个GPU创建该阶段的worker函数
                    stage_workers = []
                    for gpu_key in sorted_gpus:
                        if gpu_key not in stage_tasks_by_gpu:
                            continue
                        
                        gpu_tasks = stage_tasks_by_gpu[gpu_key]
                        worker_name = f"run_stage_{stage}_{gpu_key}".replace(',', '_').replace('-', '_')
                        stage_workers.append(worker_name)
                        
                        f.write(f"{worker_name}() {{\n")
                        f.write(f"    echo '🚀 [Worker {gpu_key}] {stage_name} 启动'\n")
                        
                        # attack_train 阶段允许并行
                        if stage == 'attack_train' and num_parallel_threads and num_parallel_threads > 1:
                            # 将任务分组并行执行
                            chunks = [gpu_tasks[i:i + num_parallel_threads] 
                                     for i in range(0, len(gpu_tasks), num_parallel_threads)]
                        else:
                            # 其他阶段串行执行
                            chunks = [[t] for t in gpu_tasks]
                        
                        for i, chunk in enumerate(chunks, 1):
                            f.write(f"    # --- Group {i}/{len(chunks)} ---\n")
                            f.write("    pids=()\n")
                            
                            for task in chunk:
                                # 确保环境变量正确传递到所有子进程
                                if task.gpu and task.gpu != 'cpu':
                                    env_export = f"export CUDA_VISIBLE_DEVICES={task.gpu}; "
                                else:
                                    env_export = ""
                                f.write(f"    # Task: {task.description}\n")
                                
                                cmds_chain = " && ".join([f"echo '  -> {s.name}' && {s.command}" for s in task.steps])
                                f.write(f"    ({env_export}{cmds_chain}) &\n")
                                f.write(f"    pids+=($!)\n")
                            
                            f.write(f"\n    echo '⏳ [Worker {gpu_key}] 等待第 {i} 组任务完成...'\n")
                            f.write("    for pid in \"${pids[@]}\"; do wait \"$pid\"; done\n")
                            f.write(f"    echo '✅ [Worker {gpu_key}] 第 {i} 组完成'\n\n")
                        
                        f.write(f"    echo '🎉 [Worker {gpu_key}] {stage_name} 完成'\n")
                        f.write("}\n\n")
                    
                    # 并行启动该阶段的所有worker
                    for worker_name in stage_workers:
                        f.write(f"{worker_name} &\n")
                    
                    # 等待该阶段所有worker完成（全局同步点）
                    f.write("\n# 等待该阶段所有任务完成\n")
                    f.write("wait\n")
                    f.write(f"echo '✅ {stage_name} 所有任务完成'\n\n")
                
                # 处理非MIA任务（如果有）
                if 'other' in tasks_by_stage_gpu:
                    f.write("\n# ========== 其他任务 ==========\n")
                    other_tasks_by_gpu = tasks_by_stage_gpu['other']
                    for gpu_key in sorted_gpus:
                        if gpu_key not in other_tasks_by_gpu:
                            continue
                        for task in other_tasks_by_gpu[gpu_key]:
                            # 确保环境变量正确传递到所有子进程
                            if task.gpu and task.gpu != 'cpu':
                                env_export = f"export CUDA_VISIBLE_DEVICES={task.gpu}; "
                            else:
                                env_export = ""
                            cmds_chain = " && ".join([f"echo '  -> {s.name}' && {s.command}" for s in task.steps])
                            f.write(f"({env_export}{cmds_chain})\n")
                
                f.write("\necho '🏁 所有阶段已完成。'\n")
            else:
                # 非MIA模式：原有的并行执行逻辑
                sorted_gpus = sorted(grouped_tasks_by_gpu.keys(), key=lambda x: (len(x), x))
                
                for gpu_key in sorted_gpus:
                    worker_name = f"run_worker_{gpu_key}".replace(',', '_')
                    chunks = grouped_tasks_by_gpu[gpu_key]
                    
                    f.write(f"{worker_name}() {{\n")
                    f.write(f"    echo '🚀 [Worker {gpu_key}] 启动，共 {len(chunks)} 组任务'\n")
                    
                    for i, chunk in enumerate(chunks, 1):
                        f.write(f"    # --- Group {i}/{len(chunks)} ---\n")
                        f.write("    pids=()\n")
                        
                        for task in chunk:
                            # 确保环境变量正确传递到所有子进程
                            if task.gpu and task.gpu != 'cpu':
                                env_export = f"export CUDA_VISIBLE_DEVICES={task.gpu}; "
                            else:
                                env_export = ""
                            f.write(f"    # Task: {task.description}\n")
                            
                            cmds_chain = " && ".join([f"echo '  -> {s.name}' && {s.command}" for s in task.steps])
                            f.write(f"    ({env_export}{cmds_chain}) &\n")
                            f.write(f"    pids+=($!)\n")
                        
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
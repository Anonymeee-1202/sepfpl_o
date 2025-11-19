import argparse
import torch
from Dassl.dassl.utils import set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer

import os
import math
import copy
import pickle
import numpy as np
import time
from tqdm import tqdm
from prettytable import PrettyTable
from utils.logger import init_logger_from_args

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def extend_cfg(cfg, args):
    """
    Add new config variables.
    """
    from yacs.config import CfgNode as CN

    # Factorization param
    cfg.FACTORIZATION = args.factorization
    cfg.RANK = args.rank

    # Differential privacy param
    cfg.NORM_THRESH = args.norm_thresh
    cfg.NOISE = args.noise
    # RDP (Rényi Differential Privacy) parameters
    cfg.RDP_ALPHA = getattr(args, 'rdp_alpha', 2.0)  # RDP阶数，默认2.0
    cfg.RDP_P = getattr(args, 'rdp_p', 1.1)  # sepfpl隐私预算分配参数p，默认2.0
    
    # 消融实验控制参数：根据factorization名称自动设置
    # 支持新的factorization名称：sepfpl_time_adaptive（只有时间适应）和sepfpl_hcse（只有HCSE）
    if args.factorization == 'sepfpl_time_adaptive':
        cfg.USE_HCSE = False
        cfg.USE_TIME_ADAPTIVE = True
    elif args.factorization == 'sepfpl_hcse':
        cfg.USE_HCSE = True
        cfg.USE_TIME_ADAPTIVE = False
    else:
        # 根据factorization自动判断（不再支持命令行参数显式指定）
        cfg.USE_HCSE = getattr(args, 'use_hcse', None)  # 是否使用HCSE结构熵引导的个性化增强模块，None表示自动判断
        cfg.USE_TIME_ADAPTIVE = getattr(args, 'use_time_adaptive', None)  # 是否使用时间适应的隐私分配机制，None表示自动判断

    # Config for DP_FPL
    cfg.TRAINER.NAME = 'DP_FPL'
    cfg.TRAINER.DP_FPL = CN()
    cfg.TRAINER.DP_FPL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.DP_FPL.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.DP_FPL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.ROOT = args.root # dataset path
    cfg.DATASET.USERS = args.num_users # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots # caltech101, dtd, oxford_flowers, oxford_pets, food101
    cfg.DATASET.PARTITION = args.partition # cifar10, cifar100
    cfg.DATASET.BETA = args.beta # cifar10, cifar100
    # Food101 per-class sampling ratio (keep class set unchanged, downsample items within each class)
    if hasattr(args, 'food101_sample_ratio'):
        cfg.DATASET.FOOD101_SAMPLE_RATIO = args.food101_sample_ratio
    else:
        cfg.DATASET.FOOD101_SAMPLE_RATIO = 1.0
    # CIFAR-100 per-class sampling ratio (keep class set unchanged, downsample items within each class)
    if hasattr(args, 'cifar100_sample_ratio'):
        cfg.DATASET.CIFAR100_SAMPLE_RATIO = args.cifar100_sample_ratio
    else:
        cfg.DATASET.CIFAR100_SAMPLE_RATIO = 1.0
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 6 if args.num_users == 6 else 4 # domainnet, office
    if args.useall:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    else:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.num_shots
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = 1 # local epoch
    cfg.OPTIM.LR = args.lr # learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True
    # Optional CLIP model path and cache directory
    if not hasattr(cfg.MODEL.BACKBONE, 'PATH'):
        cfg.MODEL.BACKBONE.PATH = None
    if not hasattr(cfg.MODEL.BACKBONE, 'CACHE_DIR'):
        cfg.MODEL.BACKBONE.CACHE_DIR = None
    if hasattr(args, 'clip_model_path') and args.clip_model_path:
        cfg.MODEL.BACKBONE.PATH = args.clip_model_path
    if hasattr(args, 'clip_cache_dir') and args.clip_cache_dir:
        cfg.MODEL.BACKBONE.CACHE_DIR = args.clip_cache_dir

    cfg.SEED = args.seed


def setup_cfg(args):
    cfg = get_cfg_default() # arguments list, type yacs.config.CfgNode _C from defaults.py
    extend_cfg(cfg, args) # add more arguments

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file) # load dataset

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file) # load model

    cfg.freeze()

    return cfg


def _should_enable_wandb(args):
    env_disabled = os.environ.get('WANDB_DISABLED', '').lower() in ['1', 'true', 'yes']
    if env_disabled:
        return False
    if getattr(args, 'wandb_mode', None) == 'disabled':
        return False
    return True


def _default_wandb_run_name(args):
    dataset = os.path.splitext(os.path.basename(args.dataset_config_file))[0]
    parts = [
        dataset,
        args.factorization,
        f"rank{args.rank}",
        f"noise{args.noise}",
        f"seed{args.seed}",
        f"users{args.num_users}",
    ]
    if args.task_id:
        parts.append(args.task_id.strip('[]'))
    return "-".join(parts)


def _prepare_wandb_tags(args):
    dataset = os.path.splitext(os.path.basename(args.dataset_config_file))[0]
    tags = {
        f"dataset:{dataset}",
        f"factorization:{args.factorization}",
        f"rank:{args.rank}",
        f"noise:{args.noise}",
        f"users:{args.num_users}",
    }
    if args.task_id:
        tags.add(f"task:{args.task_id}")
    if args.wandb_tags:
        user_tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
        tags.update(user_tags)
    return sorted(tags)


def init_wandb_run(args, cfg, logger):
    if not _WANDB_AVAILABLE:
        if _should_enable_wandb(args):
            logger.warning("已请求使用 Weights & Biases，但未安装 wandb 包，已自动禁用。")
        return None
    if not _should_enable_wandb(args):
        return None

    project = args.wandb_project or os.environ.get('WANDB_PROJECT', 'dp-fpl')
    entity = args.wandb_entity or os.environ.get('WANDB_ENTITY')
    group = args.wandb_group or os.environ.get('WANDB_GROUP')
    mode = args.wandb_mode or os.environ.get('WANDB_MODE', 'online')
    run_name = args.wandb_run_name or _default_wandb_run_name(args)
    wandb_dir = args.wandb_dir or os.environ.get('WANDB_DIR')
    tags = _prepare_wandb_tags(args)

    config_payload = {
        "dataset_config": args.dataset_config_file,
        "factorization": args.factorization,
        "rank": args.rank,
        "noise": args.noise,
        "rdp_alpha": args.rdp_alpha,
        "rdp_p": args.rdp_p,
        "num_users": args.num_users,
        "round": args.round,
        "seed": args.seed,
        "task_id": args.task_id,
        "norm_thresh": args.norm_thresh,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "partition": args.partition,
        "beta": args.beta,
        "n_ctx": args.n_ctx,
        "use_hcse": cfg.USE_HCSE,
        "use_time_adaptive": cfg.USE_TIME_ADAPTIVE,
    }

    init_kwargs = {
        "project": project,
        "entity": entity,
        "group": group,
        "mode": mode,
        "name": run_name,
        "dir": wandb_dir,
        "tags": tags,
        "config": config_payload,
        "settings": wandb.Settings(start_method="thread", _disable_stats=True),
    }
    # Remove None values to avoid wandb complaining
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    logger.info(f"[wandb] 正在初始化实验：project={project}, name={run_name}, mode={mode}")
    return wandb.init(**init_kwargs)

def save_checkpoint(args, epoch, local_weights, local_acc, neighbor_acc):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_dir = os.path.join(os.getcwd(), f'checkpoints/{dataset}')
    os.makedirs(save_dir, exist_ok=True)  # 创建目录
    save_filename = os.path.join(
        save_dir,
        f'{args.factorization}_{args.rank}_{args.noise}_{args.seed}_{args.num_users}.pth.tar'
    )
    state = {
        "epoch": epoch + 1,
        "local_weights": local_weights,
        "local_acc": local_acc,
        "neighbor_acc": neighbor_acc,
    }
    torch.save(state, save_filename)

def load_checkpoint(args):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_filename = os.path.join(
        os.getcwd(),
        f'/checkpoints/{dataset}/{args.factorization}_{args.rank}_{args.noise}_{args.seed}_{args.num_users}.pth.tar'
    )
    if not os.path.exists(save_filename):
        return 0, [{} for i in range(args.num_users)], [], []
    checkpoint = torch.load(save_filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    epoch = checkpoint["epoch"]
    local_weights = checkpoint["local_weights"]
    local_acc = checkpoint["local_acc"]
    neighbor_acc = checkpoint["neighbor_acc"]
    return epoch, local_weights, local_acc, neighbor_acc


def main(args):
    # ---------- 初始化日志与核心配置 ----------
    # init_logger_from_args 会在初始化时打印实验配置摘要（包括task_id、数据集、模型、参数等）
    logger = init_logger_from_args(args, log_dir='logs', log_to_file=True, log_to_console=True)
    
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # ---------- 数据划分模式：识别是否为Dirichlet场景 ----------
    dirichlet = False
    if args.dataset_config_file.split('/')[-1].split('.')[0] in ['cifar10', 'cifar100']:
        dirichlet = True

    # ---------- 联邦权重与梯度缓存初始化 ----------
    global_gradients = [None for _ in range(args.num_users)]
    local_weights = [{} for _ in range(args.num_users)]
    local_weights_g = [None for _ in range(args.num_users)]
    local_weights_l = [None for _ in range(args.num_users)]
    local_weights_u = [None for _ in range(args.num_users)]
    local_weights_v = [None for _ in range(args.num_users)]

    # ---------- 构建本地训练器并缓存初始权重 ----------
    local_trainer = build_trainer(cfg)
    wandb_run = init_wandb_run(args, cfg, logger)
    if wandb_run and args.wandb_watch and args.wandb_watch.lower() != 'off':
        watch_log = args.wandb_watch
        watch_logfreq = args.wandb_watch_logfreq
        wandb.watch(local_trainer.model, log=watch_log, log_freq=watch_logfreq, log_graph=False)
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())

    # ---------- sepfpl：为各客户端预填聚类提示 ----------
    try:
        # 判断是否需要初始化cluster_ctx（sepfpl和sepfpl_hcse需要）
        use_hcse_init = cfg.USE_HCSE
        if use_hcse_init is None:
            use_hcse_init = (args.factorization in ['sepfpl', 'sepfpl_hcse'])
        if use_hcse_init and 'prompt_learner.cluster_ctx' in initial_weights:
            for i in range(args.num_users):
                if 'prompt_learner.cluster_ctx' not in local_weights[i]:
                    local_weights[i]['prompt_learner.cluster_ctx'] = copy.deepcopy(initial_weights['prompt_learner.cluster_ctx'])
    except Exception as e:
        logger.warning(f"[Init] cluster_ctx 初始化失败（可忽略首轮由后续流程写入）：{e}")

    # ---------- 训练主流程所需的统计变量 ----------
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    local_acc_list, neighbor_acc_list, = [], []
    if args.resume == 'True':
        start_epoch, local_weights, local_acc_list, neighbor_acc_list = load_checkpoint(args)
        logger.info(f'Resume from epoch {start_epoch}')
    if start_epoch == max_epoch - 1:
        return
    if args.noise > 0:
        std = local_trainer.std / cfg.DATASET.USERS
    
    # ---------- 统计每轮最大批次数，辅助进度估计 ----------
    train_loaders = getattr(local_trainer, "fed_train_loader_x_dict", {})
    if train_loaders:
        max_batches_per_epoch = max(len(loader) for loader in train_loaders.values())
    else:
        max_batches_per_epoch = 0
    # ---------- 建立提示参数缓冲映射，统一读写入口 ----------
    key_to_buffer = {'prompt_learner.global_ctx': local_weights_g}
    # 判断是否需要HCSE（用于确定是否需要cluster_ctx）
    use_hcse_for_buffer = cfg.USE_HCSE
    if use_hcse_for_buffer is None:
        use_hcse_for_buffer = (args.factorization in ['sepfpl', 'sepfpl_hcse'])
    
    if args.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
        key_to_buffer['prompt_learner.local_ctx'] = local_weights_l
    if args.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
        key_to_buffer['prompt_learner.local_u_ctx'] = local_weights_u
        key_to_buffer['prompt_learner.local_v_ctx'] = local_weights_v
    if use_hcse_for_buffer:
        key_to_buffer['prompt_learner.cluster_ctx'] = [
            copy.deepcopy(local_weights[i]['prompt_learner.cluster_ctx'])
            if 'prompt_learner.cluster_ctx' in local_weights[i] else None
            for i in range(args.num_users)
        ]

    def get_prompt_default(key, idx):
        """根据提示参数名称返回默认权重，用于覆盖缺失值。"""
        if key == 'prompt_learner.global_ctx':
            return initial_weights['prompt_learner.global_ctx']
        fallback = local_weights[idx].get('prompt_learner.global_ctx', initial_weights['prompt_learner.global_ctx'])
        return initial_weights.get(key, fallback)

    for epoch in range(start_epoch, max_epoch): # global communication loop
        idxs_users = list(range(0,cfg.DATASET.USERS))
        
        # ---------- Epoch 初始化：计时与准确率统计器重置 ----------
        local_trainer.reset_epoch_timer()
        epoch_start_time = time.time()
        train_acc_sum = 0.0
        train_acc_count = 0
        
        # ---------- sepfpl：按轮更新差分隐私噪声标准差（时间适应隐私分配）----------
        # 如果use_time_adaptive为None，则根据factorization自动判断：sepfpl/sepfpl_time_adaptive/sepfpl_hcse默认启用
        use_time_adaptive = cfg.USE_TIME_ADAPTIVE
        if use_time_adaptive is None:
            use_time_adaptive = (args.factorization in ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse'])
        if use_time_adaptive and hasattr(local_trainer, 'update_std_for_round'):
            local_trainer.update_std_for_round(epoch)

        # ---------- 为每个客户端准备本地数据迭代器 ----------
        data_iters = []
        for idx in idxs_users:
            local_trainer.set_model_mode("train")
            loader = local_trainer.fed_train_loader_x_dict[idx]
            data_iters.append(iter(loader))
        max_batch = max_batches_per_epoch

        # 判断是否使用HCSE（在循环外部判断一次即可）
        # 如果use_hcse为None，则根据factorization自动判断：sepfpl/sepfpl_hcse默认启用
        use_hcse = cfg.USE_HCSE
        if use_hcse is None:
            use_hcse = (args.factorization in ['sepfpl', 'sepfpl_hcse'])

        # ---------- 批次循环：收集梯度并缓存客户端模型 ----------
        batch_range = tqdm(range(0, max_batch), desc=f"Epoch {epoch + 1}/{max_epoch} [Batch]", leave=False)
        for batch in batch_range:
            local_trainer.set_model_mode("train")
            # 按客户端收集梯度信息，供全局聚合与聚类阶段使用
            cluster_grads = [None for _ in idxs_users] if use_hcse else None
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(initial_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights[idx], strict=False)
                loss_summary = local_trainer.train_forward(idx=idx, train_iter=data_iters[idx], current_batch=batch, total_batches=max_batch)
                if loss_summary is not None and 'acc' in loss_summary:
                    train_acc_sum += loss_summary['acc']
                    train_acc_count += 1

                local_weight = local_trainer.model.state_dict()
                grad_global = local_trainer.model.prompt_learner.global_ctx.grad
                if grad_global is not None:
                    global_gradients[idx] = grad_global.data.clone()
                else:
                    global_gradients[idx] = torch.zeros_like(local_trainer.model.prompt_learner.global_ctx.data)
                for key, buffer in key_to_buffer.items():
                    if buffer is None or key not in local_weight:
                        continue
                    buffer[idx] = copy.deepcopy(local_weight[key])
                if use_hcse and 'prompt_learner.cluster_ctx' in local_weight:
                    if local_trainer.model.prompt_learner.cluster_ctx.grad is not None:
                        cluster_grads[idx] = local_trainer.model.prompt_learner.cluster_ctx.grad.data.clone()
                    else:
                        cluster_grads[idx] = torch.zeros_like(local_trainer.model.prompt_learner.cluster_ctx.data)

            # ---------- 计算全局梯度与（可选）聚类梯度 ----------
            aggregated_cluster_grads = None
            cluster_gradients_to_apply = None

            if use_hcse:
                per_user_global = []
                data_sizes = []
                for i in idxs_users:
                    try:
                        ds_len = len(local_trainer.fed_train_loader_x_dict[i].dataset)
                    except Exception:
                        ds_len = 1
                    data_sizes.append(max(1, ds_len))
                    per_user_global.append(global_gradients[i])
                total_size = float(sum(data_sizes))
                weights = [s / total_size for s in data_sizes]
                avg_global_gradient = sum(w * g for w, g in zip(weights, per_user_global))

                try:
                    from hcse.encoding_tree import (
                        PartitionTree,
                        compute_gradient_similarity_matrix_torch,
                        aggregate_gradients_by_encoding_tree,
                    )

                    for i in range(len(cluster_grads)):
                        if cluster_grads[i] is None:
                            if 'prompt_learner.cluster_ctx' in local_trainer.model.state_dict():
                                zshape = local_trainer.model.prompt_learner.cluster_ctx.data.shape
                                device = local_trainer.model.prompt_learner.cluster_ctx.data.device
                                cluster_grads[i] = torch.zeros(zshape, device=device)
                            else:
                                cluster_grads[i] = torch.zeros_like(local_trainer.model.prompt_learner.global_ctx.data)

                    sim_mat = compute_gradient_similarity_matrix_torch(cluster_grads, normalize=True)
                    # ========== top-k 稀疏与对称化 ==========
                    k = args.sepfpl_topk if hasattr(args, 'sepfpl_topk') and args.sepfpl_topk is not None else 5
                    with torch.no_grad():
                        sim_proc = sim_mat.clone()
                        n = sim_proc.shape[0]
                        for r in range(n):
                            vals, idxs = torch.topk(sim_proc[r], k=k, largest=True)
                            mask = torch.zeros_like(sim_proc[r], dtype=torch.bool)
                            mask[idxs] = True
                            mask[r] = True
                            sim_proc[r][~mask] = 0.0
                        sim_proc = torch.maximum(sim_proc, sim_proc.t())
                        sim_proc = torch.exp(sim_proc)
                    adj_matrix = sim_proc.detach().cpu().numpy()
                    tree = PartitionTree(adj_matrix)
                    tree.build_encoding_tree(k=2, mode='v2')
                    aggregated_cluster_grads = aggregate_gradients_by_encoding_tree(tree, cluster_grads, adj_matrix)

                    # 聚类梯度沿用全局学习率，无需单独eta
                    cluster_gradients_to_apply = {
                        i: aggregated_cluster_grads[i] for i in idxs_users if aggregated_cluster_grads[i] is not None
                    }
                except Exception as e:
                    logger.warning(f"[HCSE] 聚类与聚合出现异常，跳过本步: {e}")
            else:
                avg_global_gradient = sum(global_gradients) / cfg.DATASET.USERS
            if args.noise > 0:
                noise = torch.normal(0, std, size=avg_global_gradient.shape, device=avg_global_gradient.device)
                avg_global_gradient += noise
                

            # ---------- 回传与本地权重缓存同步 ----------
            for idx in idxs_users:
                for key, buffer in key_to_buffer.items():
                    should_handle = (
                        key == 'prompt_learner.global_ctx'
                        or key in local_weights[idx]
                        or key in initial_weights
                    )
                    if not should_handle or buffer is None:
                        continue
                    if not isinstance(buffer[idx], torch.Tensor):
                        buffer[idx] = copy.deepcopy(get_prompt_default(key, idx))
                local_weights[idx][key] = buffer[idx]
                local_trainer.model.load_state_dict(local_weights[idx], strict=False)

                cluster_grad_for_idx = None
                if cluster_gradients_to_apply is not None and idx in cluster_gradients_to_apply:
                    cluster_grad_for_idx = cluster_gradients_to_apply[idx]

                local_trainer.train_backward(
                    avg_global_gradient=avg_global_gradient,
                    aggregated_cluster_gradient=cluster_grad_for_idx,
                )

                local_weight = local_trainer.model.state_dict()
                for key, buffer in key_to_buffer.items():
                    if buffer is None or key not in local_weight:
                        continue
                    copied = copy.deepcopy(local_weight[key])
                    buffer[idx] = copied
            
            # 更新进度条信息（显示平均准确率）
            if train_acc_count > 0:
                avg_acc = train_acc_sum / train_acc_count
                batch_range.set_postfix({'avg_acc': f'{avg_acc:.2f}%'})
            else:
                batch_range.set_postfix({'avg_acc': 'N/A'})


        # ---------- Epoch 训练日志输出 ----------
        train_stage_end = time.time()
        train_stage_duration = train_stage_end - epoch_start_time
        avg_train_acc = (train_acc_sum / train_acc_count) if train_acc_count > 0 else 0.0
        logger.info(
            f"[Epoch {epoch + 1}/{max_epoch}] 训练耗时: {train_stage_duration:.2f}s | 平均训练准确率: {avg_train_acc:.2f}"
        )
        if wandb_run:
            wandb.log(
                {
                    "train/epoch": epoch + 1,
                    "train/avg_acc": avg_train_acc,
                    "train/duration_sec": train_stage_duration,
                },
                step=epoch + 1,
            )

        # ---------- 判断是否进入测试阶段 ----------
        should_test = False
        if max_epoch < 20:
            # 如果round总数小于20，只在最后1个epoch测试邻居
            should_test = (epoch == max_epoch - 1)
        else:
            # 当round数 >= 20，仅在最后 x = n/10 个 epoch 进行邻居测试
            last_epochs = max(1, math.ceil(max_epoch / 10))
            should_test = (epoch >= max_epoch - last_epochs)

        if should_test:
            # ---------- 测试阶段：同步权重并评估客户端 ----------
            test_start_time = time.time()
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"                    TEST START - Epoch {epoch + 1}/{max_epoch}")
            logger.info("=" * 80)
            logger.info("")
            local_trainer.set_model_mode("eval")
            results_local = []
            results_neighbor = [] if not dirichlet else None
            
            # 使用进度条显示测试进度
            test_range = tqdm(idxs_users, desc=f"Epoch {epoch + 1}/{max_epoch} [Test]", leave=False)
            for idx in test_range:
                for key, buffer in key_to_buffer.items():
                    if buffer is None:
                        continue
                    value = buffer[idx]
                    if value is None:
                        continue
                    if key != 'prompt_learner.global_ctx' and key not in local_weights[idx]:
                        continue
                    local_weights[idx][key] = value
                local_trainer.model.load_state_dict(local_weights[idx], strict=False)

                results_local.append(local_trainer.test(idx=idx, split='local'))
                if results_neighbor is not None:
                    results_neighbor.append(local_trainer.test(idx=idx, split='neighbor'))
                
                # 实时更新进度条信息
                local_acc = [res[0] for res in results_local]
                avg_local_acc = sum(local_acc) / len(local_acc) if local_acc else 0.0
                
                postfix_dict = {'local_acc': f'{avg_local_acc:.2f}%'}
                if results_neighbor is not None and len(results_neighbor) > 0:
                    neighbor_acc = [res[0] for res in results_neighbor]
                    avg_neighbor_acc = sum(neighbor_acc) / len(neighbor_acc) if neighbor_acc else 0.0
                    postfix_dict['neighbor_acc'] = f'{avg_neighbor_acc:.2f}%'
                else:
                    postfix_dict['neighbor_acc'] = 'N/A'
                
                test_range.set_postfix(postfix_dict)

            # 格式化并输出 Local 测试结果表格
            def format_results_table(results_list, title, client_ids):
                """使用 PrettyTable 格式化测试结果为表格"""
                if not results_list:
                    return ""
                
                # 确定指标数量（取第一个结果的长度）
                num_metrics = len(results_list[0])
                metric_names = ['Accuracy', 'Error Rate', 'Macro F1']
                if num_metrics > 3:
                    metric_names.extend([f'Metric {i+4}' for i in range(num_metrics - 3)])
                
                # 创建表格
                table = PrettyTable()
                table.field_names = ['Client'] + metric_names[:num_metrics]
                table.align['Client'] = 'l'
                for name in metric_names[:num_metrics]:
                    table.align[name] = 'r'
                
                # 添加每个客户端的数据
                for idx, res in enumerate(results_list):
                    client_id = client_ids[idx] if idx < len(client_ids) else idx
                    row = [f'Client {client_id}'] + [f'{val:.2f}' for val in res[:num_metrics]]
                    table.add_row(row)
                
                # 添加平均值行
                avg_values = []
                for i in range(num_metrics):
                    avg_val = sum([res[i] for res in results_list]) / len(results_list) if results_list else 0.0
                    avg_values.append(avg_val)
                avg_row = ['Average'] + [f'{val:.2f}' for val in avg_values]
                table.add_row(avg_row)
                
                # 组合为完整表格字符串
                table_str = f"\n{title}\n{table.get_string()}\n"
                return table_str
            
            # 输出 Local 测试结果表格
            local_table = format_results_table(results_local, "================= Local Test Results ==================", idxs_users)
            logger.info(local_table)
            
            local_acc = [res[0] for res in results_local]
            avg_local_acc = sum(local_acc) / len(local_acc) if local_acc else 0.0
            local_acc_list.append(avg_local_acc)

            # 输出 Neighbor 测试结果表格（如果适用）
            if results_neighbor is not None:
                neighbor_table = format_results_table(results_neighbor, "================= Neighbor Test Results ==================", idxs_users)
                logger.info(neighbor_table)
                
                neighbor_acc = [res[0] for res in results_neighbor]
                avg_neighbor_acc = sum(neighbor_acc) / len(neighbor_acc) if neighbor_acc else 0.0
                neighbor_acc_list.append(avg_neighbor_acc)
            

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"                    TEST FINISH - Epoch {epoch + 1}/{max_epoch}")
            test_duration = time.time() - test_start_time
            logger.info(f"                    测试耗时: {test_duration:.2f}s")
            logger.info("=" * 80)
            logger.info("")
            if wandb_run:
                log_payload = {
                    "test/epoch": epoch + 1,
                    "test/local_acc_avg": avg_local_acc,
                    "test/duration_sec": test_duration,
                }
                if results_neighbor is not None:
                    log_payload["test/neighbor_acc_avg"] = avg_neighbor_acc
                wandb.log(log_payload, step=epoch + 1)

            # ---------- 保存训练曲线与权重检查点 ----------
            save_checkpoint(args, epoch, local_weights, local_acc_list, neighbor_acc_list)
            dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
            output_dir = os.path.join(os.getcwd(), f'outputs/{dataset_name}')
            os.makedirs(output_dir, exist_ok=True)
            pickle.dump(
                [local_acc_list, neighbor_acc_list],
                open(
                    os.path.join(
                        output_dir,
                        f'acc_{args.factorization}_{args.rank}_{args.noise}_{args.seed}_{args.num_users}.pkl'
                    ),
                    'wb'
                )
            )
    if local_acc_list:
        max_local = max(local_acc_list)
        mean_local = np.mean(local_acc_list[-5:]) if len(local_acc_list) >= 5 else np.mean(local_acc_list)
        logger.info(f"maximum test local acc: {max_local:.3f}")
        logger.info(f"mean of local acc: {mean_local:.3f}")
    else:
        max_local = None
        mean_local = None
    if not dirichlet and neighbor_acc_list:
        max_neighbor = max(neighbor_acc_list)
        mean_neighbor = np.mean(neighbor_acc_list[-5:]) if len(neighbor_acc_list) >= 5 else np.mean(neighbor_acc_list)
        logger.info(f"maximum test neighbor acc: {max_neighbor:.3f}")
        logger.info(f"mean of neighbor acc: {mean_neighbor:.3f}")
    else:
        max_neighbor = None
        mean_neighbor = None

    if wandb_run:
        summary_updates = {}
        if max_local is not None:
            summary_updates["summary/local_acc_max"] = max_local
        if mean_local is not None:
            summary_updates["summary/local_acc_mean"] = mean_local
        if max_neighbor is not None:
            summary_updates["summary/neighbor_acc_max"] = max_neighbor
        if mean_neighbor is not None:
            summary_updates["summary/neighbor_acc_mean"] = mean_neighbor
        wandb_run.summary.update(summary_updates)
        wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=20, help="number of communication round")
    parser.add_argument('--num-users', type=int, default=10, help="number of users")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-batch-size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test-batch-size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    # parameters of factorization and differential privacy
    parser.add_argument('--factorization', type=str, default='dpfpl', help='Choose from: promptfl, fedotp, fedpgp, dplora, dpfpl, sepfpl, sepfpl_time_adaptive, sepfpl_hcse')
    parser.add_argument('--rank', type=int, default=8, help='matrix factorization rank')
    parser.add_argument('--norm-thresh', type=float, default=10.0, help='clipping norm threshold')
    parser.add_argument('--noise', type=float, default=0.0, help='differential privacy noise scale')
    parser.add_argument('--rdp-alpha', type=float, default=2.0, help='RDP (Rényi Differential Privacy) order alpha, default 2.0')
    parser.add_argument('--rdp-p', type=float, default=1.05, help='RDP privacy budget allocation parameter p for sepfpl, default 2.0')

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num-shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=True, help="is useall, True for all training samples, False for few shot learning")
    parser.add_argument('--food101-sample-ratio', type=float, default=0.1, help="per-class sampling ratio for Food101 (0,1], keep class set unchanged")
    parser.add_argument('--cifar100-sample-ratio', type=float, default=0.2, help="per-class sampling ratio for CIFAR-100 (0,1], keep class set unchanged")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.3, help='The parameter for the dirichlet distribution for data partitioning')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    # sepfpl-specific optional params
    parser.add_argument('--sepfpl-topk', type=int, default=8, help='top-k neighbors for HCSE graph sparsification (sepfpl only)')

    # parameters of path
    parser.add_argument("--root", type=str, default="/datasets", help="path to dataset")
    parser.add_argument("--config-file", type=str, default="configs/trainers/DP-FPL/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar100.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default="False", help="resume training or not")
    parser.add_argument('--task-id', type=str, default=None, help='任务编号标识，格式如 "[1/100]" (用于日志标识)')
    # Optional CLIP model parameters
    parser.add_argument('--clip-model-path', type=str, default=None, help='path to local CLIP model file (optional)')
    parser.add_argument('--clip-cache-dir', type=str, default=None, help='directory to cache/download CLIP models (optional)')

    # wandb logging configuration
    default_wandb_mode = os.environ.get('WANDB_MODE', 'online' if _WANDB_AVAILABLE else 'disabled')
    parser.add_argument('--wandb-mode', type=str, default=default_wandb_mode, choices=['online', 'offline', 'disabled'],
                        help='wandb 日志模式：online/offline/disabled')
    parser.add_argument('--wandb-project', type=str, default=None, help='wandb 项目名称（默认 dp-fpl）')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity/团队名称')
    parser.add_argument('--wandb-group', type=str, default=None, help='wandb group，用于实验分组')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb run 名称（默认根据实验参数自动生成）')
    parser.add_argument('--wandb-dir', type=str, default=None, help='wandb 本地缓存目录')
    parser.add_argument('--wandb-tags', type=str, default=None, help='额外的 wandb 标签，使用逗号分隔')
    parser.add_argument('--wandb-watch', type=str, default='gradients', help='wandb.watch log 类型，设置为off禁用')
    parser.add_argument('--wandb-watch-logfreq', type=int, default=200, help='wandb.watch 日志频率')

    args = parser.parse_args()
    
    main(args)


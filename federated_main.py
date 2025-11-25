import argparse
import os
import math
import copy
import pickle
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Union

from tqdm import tqdm
from prettytable import PrettyTable
from yacs.config import CfgNode

# Dassl 依赖
from Dassl.dassl.utils import set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer, TrainerX

# 本地工具
from utils.logger import init_logger_from_args

# WandB 处理
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


# ==============================================================================
# 配置与路径管理
# ==============================================================================

def extend_cfg(cfg: CfgNode, args: argparse.Namespace):
    """扩展 Dassl 默认配置，注入本项目额外需要的配置项。"""
    from yacs.config import CfgNode as CN

    # --- 矩阵分解与隐私参数 ---
    cfg.FACTORIZATION = args.factorization
    cfg.RANK = args.rank
    cfg.NORM_THRESH = args.norm_thresh
    cfg.NOISE = args.noise
    cfg.RDP_ALPHA = getattr(args, 'rdp_alpha', 2.0)
    cfg.RDP_P = getattr(args, 'rdp_p', 1.1)

    # --- Trainer 配置: DP_FPL ---
    cfg.TRAINER.NAME = 'DP_FPL'
    cfg.TRAINER.DP_FPL = CN()
    cfg.TRAINER.DP_FPL.N_CTX = args.n_ctx
    cfg.TRAINER.DP_FPL.PREC = "fp32"
    cfg.TRAINER.DP_FPL.CLASS_TOKEN_POSITION = "end"

    # --- 数据集配置 ---
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.USERS = args.num_users
    cfg.DATASET.IID = args.iid
    cfg.DATASET.USEALL = args.useall
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.FOOD101_SAMPLE_RATIO = getattr(args, 'food101_sample_ratio', 1.0)
    cfg.DATASET.CIFAR100_SAMPLE_RATIO = getattr(args, 'cifar100_sample_ratio', 1.0)

    # Domain 设置
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 6 if args.num_users == 6 else 4

    # Batch Size
    if args.useall:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    else:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.num_shots
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # --- 优化器与模型 ---
    cfg.OPTIM.ROUND = args.round
    cfg.OPTIM.MAX_EPOCH = 1
    cfg.OPTIM.LR = args.lr
    cfg.MODEL.BACKBONE.PRETRAINED = True
    
    # CLIP 路径处理
    if not hasattr(cfg.MODEL.BACKBONE, 'PATH'):
        cfg.MODEL.BACKBONE.PATH = None
    if not hasattr(cfg.MODEL.BACKBONE, 'CACHE_DIR'):
        cfg.MODEL.BACKBONE.CACHE_DIR = None
    
    if getattr(args, 'clip_model_path', None):
        cfg.MODEL.BACKBONE.PATH = args.clip_model_path
    if getattr(args, 'clip_cache_dir', None):
        cfg.MODEL.BACKBONE.CACHE_DIR = args.clip_cache_dir

    cfg.SEED = args.seed


def setup_cfg(args: argparse.Namespace) -> CfgNode:
    """构造并返回最终配置。"""
    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def get_paths(args: argparse.Namespace):
    """集中生成文件保存路径。"""
    wandb_group = getattr(args, 'wandb_group', None) or 'default'
    dataset_name = os.path.splitext(os.path.basename(args.dataset_config_file))[0]
    base_dir = os.path.expanduser('~/data/sepfpl')
    
    ckpt_dir = os.path.join(base_dir, 'checkpoints', wandb_group, dataset_name)
    output_dir = os.path.join(base_dir, 'outputs', wandb_group, dataset_name)
    
    file_id = f'{args.factorization}_{args.rank}_{args.noise}_{args.seed}_{args.num_users}'
    ckpt_path = os.path.join(ckpt_dir, f'{file_id}.pth.tar')
    pkl_path = os.path.join(output_dir, f'acc_{file_id}.pkl')
    
    return ckpt_path, pkl_path, dataset_name


# ==============================================================================
# 联邦状态管理 (State Management)
# ==============================================================================

class FederatedStateManager:
    """管理联邦学习过程中的所有权重、梯度缓存和特定组件的状态。"""
    def __init__(self, num_users: int, initial_model_state: Dict[str, torch.Tensor], factorization: str):
        self.num_users = num_users
        self.initial_weights = copy.deepcopy(initial_model_state)
        self.factorization = factorization

        # 核心权重与梯度
        self.local_weights: List[Dict[str, torch.Tensor]] = [{} for _ in range(num_users)]
        self.global_gradients: List[Optional[torch.Tensor]] = [None for _ in range(num_users)]
        
        # HCSE (Hierarchical Clustering) 特定
        self.use_hcse = factorization in ('sepfpl', 'sepfpl_hcse')
        self.cluster_grads: Optional[List[Optional[torch.Tensor]]] = [None for _ in range(num_users)] if self.use_hcse else None
        
        # 初始化 cluster_ctx (如果存在)
        if self.use_hcse and 'prompt_learner.cluster_ctx' in self.initial_weights:
            for i in range(num_users):
                self.local_weights[i]['prompt_learner.cluster_ctx'] = copy.deepcopy(
                    self.initial_weights['prompt_learner.cluster_ctx']
                )

        # Buffer 映射表初始化
        self.buffers: Dict[str, List[Any]] = {
            'prompt_learner.global_ctx': [None for _ in range(num_users)]
        }
        
        # 根据算法变体注册不同的 Buffer
        if factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
            self.buffers['prompt_learner.local_ctx'] = [None for _ in range(num_users)]
        
        if factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
            self.buffers['prompt_learner.local_u_ctx'] = [None for _ in range(num_users)]
            self.buffers['prompt_learner.local_v_ctx'] = [None for _ in range(num_users)]
            
        if self.use_hcse:
             self.buffers['prompt_learner.cluster_ctx'] = [
                copy.deepcopy(self.local_weights[i]['prompt_learner.cluster_ctx']) 
                if 'prompt_learner.cluster_ctx' in self.local_weights[i] else None
                for i in range(num_users)
            ]

    def get_prompt_default(self, key: str, idx: int) -> torch.Tensor:
        """获取权重的默认值 (fallback)。"""
        if key == 'prompt_learner.global_ctx':
            return self.initial_weights['prompt_learner.global_ctx']
        
        # 尝试回退到该客户端的 global_ctx，最后回退到初始权重
        fallback = self.local_weights[idx].get('prompt_learner.global_ctx', self.initial_weights['prompt_learner.global_ctx'])
        return self.initial_weights.get(key, fallback)

    def sync_buffers_from_local(self, idx: int, current_model_state: Dict[str, torch.Tensor]):
        """将客户端本地训练后的权重同步到 buffer 中。"""
        for key, buffer_list in self.buffers.items():
            if key in current_model_state:
                buffer_list[idx] = copy.deepcopy(current_model_state[key])

    def sync_local_from_buffers(self, idx: int):
        """将 buffer 中的权重同步回客户端本地权重字典。"""
        for key, buffer_list in self.buffers.items():
            val = buffer_list[idx]
            
            # 只有 global_ctx 是必须存在的，其他如果是 None 可以跳过 (除非它是必须的初始化)
            should_handle = (
                key == 'prompt_learner.global_ctx' or 
                key in self.local_weights[idx] or 
                key in self.initial_weights
            )

            if not should_handle or val is None:
                continue

            # 如果 buffer 里的值不是 Tensor (可能是初始 None)，则填充默认值
            if not isinstance(val, torch.Tensor):
                val = copy.deepcopy(self.get_prompt_default(key, idx))
                buffer_list[idx] = val # 更新 buffer 以备后用

            self.local_weights[idx][key] = val


# ==============================================================================
# 梯度聚合与隐私处理 (Aggregator)
# ==============================================================================

class GradientAggregator:
    """处理梯度的聚合、聚类 (HCSE) 和差分隐私噪声添加。"""
    
    @staticmethod
    def aggregate(
        args: argparse.Namespace, 
        cfg: CfgNode, 
        state: FederatedStateManager, 
        trainer: TrainerX,
        logger
    ) -> Tuple[torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
        """
        返回: (avg_global_gradient, cluster_gradients_to_apply)
        """
        use_hcse = state.use_hcse
        users = list(range(cfg.DATASET.USERS))
        
        # 1. 计算全局梯度的加权平均
        if use_hcse:
            # HCSE 模式下基于数据量加权
            per_user_global = []
            data_sizes = []
            for i in users:
                try:
                    ds_len = len(trainer.fed_train_loader_x_dict[i].dataset)
                except Exception:
                    ds_len = 1
                data_sizes.append(max(1, ds_len))
                per_user_global.append(state.global_gradients[i])
            
            total_size = float(sum(data_sizes))
            weights = [s / total_size for s in data_sizes]
            avg_global_gradient = sum(w * g for w, g in zip(weights, per_user_global))
        else:
            # 普通平均
            avg_global_gradient = sum(state.global_gradients) / cfg.DATASET.USERS

        # 2. HCSE 聚类梯度处理 (如果启用)
        cluster_gradients_to_apply = None
        if use_hcse:
            cluster_gradients_to_apply = GradientAggregator._process_hcse(
                args, state, trainer, users, logger
            )

        # 3. 差分隐私噪声添加 (DP)
        if args.noise > 0:
            std = trainer.std / cfg.DATASET.USERS
            noise = torch.normal(
                0, std, 
                size=avg_global_gradient.shape, 
                device=avg_global_gradient.device
            )
            avg_global_gradient += noise

        return avg_global_gradient, cluster_gradients_to_apply

    @staticmethod
    def _process_hcse(args, state, trainer, users, logger):
        """处理 HCSE 相关的相似度计算与树聚合。"""
        cluster_grads = state.cluster_grads
        
        # 填充缺失梯度
        for i in range(len(cluster_grads)):
            if cluster_grads[i] is None:
                # 尝试获取 shape 和 device
                if 'prompt_learner.cluster_ctx' in trainer.model.state_dict():
                    ref = trainer.model.prompt_learner.cluster_ctx.data
                    cluster_grads[i] = torch.zeros(ref.shape, device=ref.device)
                else:
                    ref = trainer.model.prompt_learner.global_ctx.data
                    cluster_grads[i] = torch.zeros_like(ref)

        try:
            from hcse.encoding_tree import (
                PartitionTree,
                compute_gradient_similarity_matrix_torch,
                aggregate_gradients_by_encoding_tree,
            )

            # 计算相似度矩阵
            sim_mat = compute_gradient_similarity_matrix_torch(cluster_grads, normalize=True)

            # 稀疏化与处理
            k = getattr(args, 'sepfpl_topk', 5) or 5
            with torch.no_grad():
                sim_proc = sim_mat.clone()
                n = sim_proc.shape[0]
                for r in range(n):
                    vals, idxs = torch.topk(sim_proc[r], k=k, largest=True)
                    mask = torch.zeros_like(sim_proc[r], dtype=torch.bool)
                    mask[idxs] = True
                    mask[r] = True
                    sim_proc[r][~mask] = 0.0
                
                # 对称化与指数映射
                sim_proc = torch.maximum(sim_proc, sim_proc.t())
                sim_proc = torch.exp(sim_proc)

            adj_matrix = sim_proc.detach().cpu().numpy()
            
            # 构建编码树与聚合
            tree = PartitionTree(adj_matrix)
            tree.build_encoding_tree(k=2, mode='v2')
            
            aggregated_cluster_grads = aggregate_gradients_by_encoding_tree(
                tree, cluster_grads, adj_matrix
            )

            return {
                i: aggregated_cluster_grads[i] 
                for i in users 
                if aggregated_cluster_grads[i] is not None
            }

        except Exception as e:
            logger.warning(f"[HCSE] 聚类聚合过程异常，将跳过本轮聚类更新: {e}")
            return None


# ==============================================================================
# WandB 辅助工具
# ==============================================================================

def init_wandb(args, cfg, logger):
    """初始化 WandB。"""
    disabled = os.environ.get('WANDB_DISABLED', '').lower() in ['1', 'true', 'yes']
    if disabled or not _WANDB_AVAILABLE:
        return None

    project = 'SepFPL'
    run_name = f"{os.path.splitext(os.path.basename(args.dataset_config_file))[0]}-{args.factorization}-rank{args.rank}-noise{args.noise}-seed{args.seed}-users{args.num_users}"
    if args.task_id:
        run_name += f"-{args.task_id.strip('[]')}"

    tags = [
        f"ds:{os.path.basename(args.dataset_config_file)}",
        f"algo:{args.factorization}",
        f"noise:{args.noise}"
    ]

    config_payload = {k: v for k, v in vars(args).items() if v is not None}
    
    logger.info(f"[wandb] Init run: {run_name}")
    return wandb.init(
        project=project,
        group=args.wandb_group,
        mode='offline',
        name=run_name,
        tags=tags,
        config=config_payload,
        dir=os.path.expanduser('~/data/sepfpl/'),
        settings=wandb.Settings(start_method="thread", _disable_stats=True)
    )

# ==============================================================================
# 核心逻辑：训练与测试
# ==============================================================================

def evaluate_epoch(
    args, 
    epoch: int, 
    max_epoch: int, 
    trainer: TrainerX, 
    state: FederatedStateManager, 
    is_dirichlet: bool, 
    logger, 
    wandb_run
) -> Tuple[float, Optional[float]]:
    """执行一轮测试并返回 (avg_local_acc, avg_neighbor_acc)。"""
    
    show_details = getattr(args, "test_verbose_log", False)
    if show_details:
        logger.info(f"\n{'='*30} TEST START - Epoch {epoch+1}/{max_epoch} {'='*30}\n")
    
    trainer.set_model_mode("eval")
    results_local = []
    results_neighbor = [] if not is_dirichlet else None

    # 测试进度条
    users = range(state.num_users)
    pbar = tqdm(users, desc=f"Test Ep {epoch+1}", leave=False)
    
    for idx in pbar:
        # 同步状态到模型
        state.sync_local_from_buffers(idx)
        trainer.model.load_state_dict(state.local_weights[idx], strict=False)

        # 1. Local Test
        res_loc = trainer.test(idx=idx, split='local')
        results_local.append(res_loc)

        # 2. Neighbor Test (Optional)
        if results_neighbor is not None:
            res_nei = trainer.test(idx=idx, split='neighbor')
            results_neighbor.append(res_nei)

        # 更新进度条信息
        avg_loc = np.mean([r[0] for r in results_local])
        postfix = {'local': f'{avg_loc:.2f}%'}
        if results_neighbor:
            avg_nei = np.mean([r[0] for r in results_neighbor])
            postfix['neighbor'] = f'{avg_nei:.2f}%'
        pbar.set_postfix(postfix)

    # 结果计算
    avg_local_acc = np.mean([r[0] for r in results_local])
    avg_neighbor_acc = np.mean([r[0] for r in results_neighbor]) if results_neighbor else None

    # 打印表格
    def log_table(res_list, title):
        if not res_list: return
        t = PrettyTable(['Client', 'Acc', 'Err', 'F1'])
        t.align = 'r'; t.align['Client'] = 'l'
        for i, r in enumerate(res_list):
            t.add_row([f'C-{i}'] + [f'{x:.2f}' for x in r[:3]])
        t.add_row(['AVG'] + [f'{np.mean([r[k] for r in res_list]):.2f}' for k in range(3)])
        if show_details: logger.info(f"\n{title}\n{t}")

    log_table(results_local, "--- Local Results ---")
    if results_neighbor:
        log_table(results_neighbor, "--- Neighbor Results ---")

    # 日志与 WandB
    log_msg = f"[Epoch {epoch+1}] Test | Local: {avg_local_acc:.2f}%"
    if avg_neighbor_acc is not None:
        log_msg += f" | Neighbor: {avg_neighbor_acc:.2f}%"
    logger.info(log_msg)

    if wandb_run:
        log_dict = {
            "test/epoch": epoch + 1,
            "test/local_acc": avg_local_acc
        }
        if avg_neighbor_acc is not None:
            log_dict["test/neighbor_acc"] = avg_neighbor_acc
        wandb_run.log(log_dict, step=epoch + 1)

    return avg_local_acc, avg_neighbor_acc


def save_checkpoint_data(args, epoch, state, acc_history, ckpt_path, pkl_path):
    """保存模型 Checkpoint 和精度历史。"""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    # 保存权重
    torch.save({
        "epoch": epoch + 1,
        "local_weights": state.local_weights,
        "local_acc": acc_history[0],
        "neighbor_acc": acc_history[1],
    }, ckpt_path)

    # 保存精度曲线 Pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(acc_history, f)


def load_checkpoint_if_needed(args, ckpt_path, logger):
    """尝试加载 Checkpoint。"""
    if args.resume != 'True' or not os.path.exists(ckpt_path):
        return 0, None, [], []
    
    logger.info(f"Resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return checkpoint["epoch"], checkpoint["local_weights"], checkpoint["local_acc"], checkpoint["neighbor_acc"]


def main(args):
    # --- 初始化 ---
    logger = init_logger_from_args(args, log_dir=os.path.expanduser('~/data/sepfpl/logs'), log_to_file=True, log_to_console=True)
    cfg = setup_cfg(args)
    if cfg.SEED >= 0: set_random_seed(cfg.SEED)
    if torch.cuda.is_available() and cfg.USE_CUDA: torch.backends.cudnn.benchmark = True

    # 路径准备
    ckpt_path, pkl_path, dataset_name = get_paths(args)
    dirichlet = dataset_name in ['cifar10', 'cifar100']

    # 构建 Trainer
    local_trainer = build_trainer(cfg)
    wandb_run = init_wandb(args, cfg, logger)
    if wandb_run:
        wandb.watch(local_trainer.model, log='gradients', log_freq=200, log_graph=False)

    # 状态管理初始化
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())
    state = FederatedStateManager(args.num_users, initial_weights, args.factorization)
    
    # 恢复训练检查
    start_epoch, loaded_weights, local_acc_list, neighbor_acc_list = load_checkpoint_if_needed(args, ckpt_path, logger)
    if loaded_weights:
        state.local_weights = loaded_weights
    
    if start_epoch >= cfg.OPTIM.ROUND:
        logger.info("Training already completed.")
        return

    # 预计算
    max_batches = max(len(l) for l in local_trainer.fed_train_loader_x_dict.values()) if local_trainer.fed_train_loader_x_dict else 0
    
    # --- 主循环 ---
    for epoch in range(start_epoch, cfg.OPTIM.ROUND):
        epoch_start = time.time()
        local_trainer.reset_epoch_timer()
        
        # 自适应噪声更新
        if args.factorization in ('sepfpl', 'sepfpl_time_adaptive') and hasattr(local_trainer, 'update_std_for_round'):
            local_trainer.update_std_for_round(epoch)

        # 准备数据迭代器
        data_iters = {i: iter(local_trainer.fed_train_loader_x_dict[i]) for i in range(args.num_users)}
        
        # === 训练阶段 ===
        train_acc_meter = []
        pbar = tqdm(range(max_batches), desc=f"Ep {epoch+1}/{cfg.OPTIM.ROUND} [Train]", leave=False)
        
        for batch_idx in pbar:
            local_trainer.set_model_mode("train")
            
            # --- 1. 客户端本地更新 ---
            for i in range(args.num_users):
                # 加载权重 (Epoch 0 加载初始，否则加载上一轮本地)
                w_init = state.initial_weights if epoch == 0 else state.local_weights[i]
                local_trainer.model.load_state_dict(w_init, strict=False)

                # 前向传播
                loss_info = local_trainer.train_forward(
                    idx=i, train_iter=data_iters[i], current_batch=batch_idx, total_batches=max_batches
                )
                if loss_info and 'acc' in loss_info:
                    train_acc_meter.append(loss_info['acc'])

                # 提取梯度与状态
                current_w = local_trainer.model.state_dict()
                
                # Global Gradient
                g_grad = local_trainer.model.prompt_learner.global_ctx.grad
                state.global_gradients[i] = g_grad.data.clone() if g_grad is not None else torch.zeros_like(local_trainer.model.prompt_learner.global_ctx.data)
                
                # Cluster Gradient (HCSE)
                if state.use_hcse:
                    c_grad = getattr(local_trainer.model.prompt_learner, 'cluster_ctx', None)
                    if c_grad is not None and c_grad.grad is not None:
                        state.cluster_grads[i] = c_grad.grad.data.clone()
                    else:
                        state.cluster_grads[i] = None # Will be filled in aggregator

                # 同步到 Buffer
                state.sync_buffers_from_local(i, current_w)

            # --- 2. 梯度聚合 (HCSE + DP) ---
            avg_grad, cluster_grads_map = GradientAggregator.aggregate(args, cfg, state, local_trainer, logger)

            # --- 3. 回传与更新 ---
            for i in range(args.num_users):
                state.sync_local_from_buffers(i) # 从 buffer 拿最新的上下文
                local_trainer.model.load_state_dict(state.local_weights[i], strict=False)

                # 选择性应用 Cluster Gradient
                c_grad_apply = cluster_grads_map.get(i) if cluster_grads_map else None
                
                local_trainer.train_backward(
                    avg_global_gradient=avg_grad,
                    aggregated_cluster_gradient=c_grad_apply
                )
                
                # 更新完成，写回 State
                state.local_weights[i] = copy.deepcopy(local_trainer.model.state_dict())
                state.sync_buffers_from_local(i, state.local_weights[i])

            # 显示进度
            if train_acc_meter:
                pbar.set_postfix({'acc': f'{np.mean(train_acc_meter):.2f}%'})

        # 记录训练日志
        train_time = time.time() - epoch_start
        logger.info(f"[Epoch {epoch+1}] Train Time: {train_time:.2f}s | Acc: {np.mean(train_acc_meter):.2f}%")
        if wandb_run:
            wandb_run.log({"train/epoch": epoch+1, "train/avg_acc": np.mean(train_acc_meter)}, step=epoch+1)

        # === 测试阶段 ===
        loc_acc, nei_acc = evaluate_epoch(args, epoch, cfg.OPTIM.ROUND, local_trainer, state, dirichlet, logger, wandb_run)
        local_acc_list.append(loc_acc)
        if nei_acc is not None:
            neighbor_acc_list.append(nei_acc)

        # === 保存 ===
        save_checkpoint_data(args, epoch, state, [local_acc_list, neighbor_acc_list], ckpt_path, pkl_path)

    # --- 结束总结 ---
    logger.info("Training Finished.")
    if local_acc_list:
        local_window = min(10, len(local_acc_list))
        local_best_avg = -float('inf')
        local_best_range = (0, local_window - 1)
        for start in range(0, len(local_acc_list) - local_window + 1):
            window_avg = sum(local_acc_list[start:start + local_window]) / local_window
            if window_avg > local_best_avg:
                local_best_avg = window_avg
                local_best_range = (start, start + local_window - 1)
        logger.info(
            f"Local best {local_window}-epoch window: Epoch {local_best_range[0] + 1}-{local_best_range[1] + 1} | "
            f"Avg Local Acc: {local_best_avg:.2f}%"
        )
        logger.info(f"Best Local Acc: {max(local_acc_list):.2f}%")
    if neighbor_acc_list:
        neighbor_window = min(10, len(neighbor_acc_list))
        neighbor_best_avg = -float('inf')
        neighbor_best_range = (0, neighbor_window - 1)
        for start in range(0, len(neighbor_acc_list) - neighbor_window + 1):
            window_avg = sum(neighbor_acc_list[start:start + neighbor_window]) / neighbor_window
            if window_avg > neighbor_best_avg:
                neighbor_best_avg = window_avg
                neighbor_best_range = (start, start + neighbor_window - 1)
        logger.info(
            f"Neighbor best {neighbor_window}-epoch window: Epoch {neighbor_best_range[0] + 1}-{neighbor_best_range[1] + 1} | "
            f"Avg Neighbor Acc: {neighbor_best_avg:.2f}%"
        )
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--round', type=int, default=20, help="Communication rounds")
    parser.add_argument('--num-users', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    
    # Method & Privacy
    parser.add_argument('--factorization', type=str, default='sepfpl')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--norm-thresh', type=float, default=10.0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--rdp-alpha', type=float, default=2.0)
    parser.add_argument('--rdp-p', type=float, default=1.01)
    
    # Data
    parser.add_argument('--iid', default=False, action='store_true') # 注意这里如果是bool通常用action
    parser.add_argument('--num-shots', type=int, default=16)
    parser.add_argument('--useall', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--food101-sample-ratio', type=float, default=0.1)
    parser.add_argument('--cifar100-sample-ratio', type=float, default=0.2)
    parser.add_argument('--partition', type=str, default='noniid-labeldir')
    parser.add_argument('--beta', type=float, default=0.3)
    
    # Prompt
    parser.add_argument('--n_ctx', type=int, default=16)
    parser.add_argument('--sepfpl-topk', type=int, default=8)
    
    # Paths & Configs
    parser.add_argument("--root", type=str, default="/datasets")
    parser.add_argument("--config-file", type=str, default="configs/trainers/DP-FPL/vit_b16.yaml")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar100.yaml")
    parser.add_argument("--resume", type=str, default="False")
    parser.add_argument('--task-id', type=str, default=None)
    parser.add_argument('--clip-model-path', type=str, default=None)
    parser.add_argument('--clip-cache-dir', type=str, default=None)
    
    # Logging
    parser.add_argument('--wandb-group', type=str, default=None)
    parser.add_argument('--test-verbose-log', action='store_true', default=False, help="Show detailed test tables")

    args = parser.parse_args()
    
    # 兼容处理 iid/useall 的 bool 解析 (防止命令行传字符串出错)
    if isinstance(args.iid, str): args.iid = args.iid.lower() == 'true'
    
    main(args)
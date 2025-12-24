"""
联邦学习主程序：支持多种矩阵分解算法和差分隐私机制

主要功能：
- 联邦提示学习（Federated Prompt Learning）
- 支持多种算法变体：sepfpl, dpfpl, fedotp, fedpgp, dplora 等
- 支持 HCSE（分层聚类结构熵）和时间自适应隐私分配
- 集成 wandb 实验跟踪
"""

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
from utils.config_utils import setup_cfg
from utils.checkpoint_utils import (
    build_filename_suffix,
    get_output_dir,
    save_checkpoint,
    load_checkpoint,
)

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _should_enable_wandb(args):
    """
    判断是否应该启用 wandb：
    - 若环境变量 WANDB_DISABLED 设置为 1/true/yes，则强制禁用。
    """
    env_disabled = os.environ.get('WANDB_DISABLED', '').lower() in ['1', 'true', 'yes']
    if env_disabled:
        return False
    return True


def _default_wandb_run_name(args):
    """
    为当前实验生成一个默认的 wandb 运行名称，便于区分。
    示例：cifar100-sepfpl-rank8-noise0.1-seed1-users10-[1/100]
    """
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
    """
    生成 wandb 用的标签列表（tags），用于过滤与对比实验。
    """
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
    return sorted(tags)


def _use_hcse(args) -> bool:
    """
    根据 factorization 名称判断当前是否启用 HCSE（Hierarchical Clustering based Structural Entropy）。

    - sepfpl / sepfpl_hcse：启用 HCSE；
    - 其他算法：默认关闭。
    """
    return args.factorization in ('sepfpl', 'sepfpl_hcse')


def _use_time_adaptive(args) -> bool:
    """
    根据 factorization 名称判断是否启用时间自适应隐私分配。

    - sepfpl / sepfpl_time_adaptive：启用时间自适应；
    - 其他算法：默认关闭。
    """
    return args.factorization in ('sepfpl', 'sepfpl_time_adaptive')


def init_wandb_run(args, cfg, logger):
    """
    初始化 wandb 运行（若可用且未被禁用）。
    
    Args:
        args: 命令行参数对象
        cfg: 配置对象
        logger: 日志记录器
        
    Returns:
        wandb.Run 对象，若不可用或已禁用则返回 None
    """
    if not _WANDB_AVAILABLE:
        if _should_enable_wandb(args):
            logger.warning("已请求使用 Weights & Biases，但未安装 wandb 包，已自动禁用。")
        return None
    if not _should_enable_wandb(args):
        return None

    project = 'SepFPL'
    group = args.wandb_group
    mode = 'offline'
    run_name = _default_wandb_run_name(args)
    tags = _prepare_wandb_tags(args)

    # 记录关键超参数，方便在 wandb 界面中查看与对比
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
    }

    init_kwargs = {
        "project": project,
        "group": group,
        "mode": mode,
        "name": run_name,
        "tags": tags,
        "config": config_payload,
        "dir": os.path.expanduser('~/code/sepfpl/'),
        "settings": wandb.Settings(start_method="thread", _disable_stats=True),
    }
    # 去除值为 None 的键，以避免 wandb 报错
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    logger.info(f"[wandb] 正在初始化实验：project={project}, name={run_name}, mode={mode}")
    return wandb.init(**init_kwargs)


def initialize_prompt_buffers(args, local_weights, use_hcse_flag):
    """
    初始化提示参数缓冲区映射。
    
    根据不同的算法变体，决定需要维护哪些提示参数缓冲区：
    - global_ctx: 所有算法都需要
    - local_ctx: fedotp, dplora, dpfpl, sepfpl 系列
    - local_u_ctx / local_v_ctx: fedpgp, dplora, dpfpl, sepfpl 系列
    - cluster_ctx: 仅 HCSE 模式需要
    
    Args:
        args: 命令行参数对象
        local_weights: 客户端权重列表
        use_hcse_flag: 是否启用 HCSE
        
    Returns:
        dict: 提示参数键到缓冲区的映射
    """
    local_weights_g = [None for _ in range(args.num_users)]
    local_weights_l = [None for _ in range(args.num_users)]
    local_weights_u = [None for _ in range(args.num_users)]
    local_weights_v = [None for _ in range(args.num_users)]
    
    key_to_buffer = {'prompt_learner.global_ctx': local_weights_g}
    
    # 根据算法变体决定维护哪些提示参数
    if args.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
        key_to_buffer['prompt_learner.local_ctx'] = local_weights_l
    if args.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
        key_to_buffer['prompt_learner.local_u_ctx'] = local_weights_u
        key_to_buffer['prompt_learner.local_v_ctx'] = local_weights_v
    if use_hcse_flag:
        key_to_buffer['prompt_learner.cluster_ctx'] = [
            copy.deepcopy(local_weights[i]['prompt_learner.cluster_ctx'])
            if 'prompt_learner.cluster_ctx' in local_weights[i] else None
            for i in range(args.num_users)
        ]
    
    return key_to_buffer


def collect_client_gradients(local_trainer, idxs_users, data_iters, batch, max_batch,
                             initial_weights, local_weights, global_gradients, 
                             key_to_buffer, use_hcse, epoch):
    """
    收集所有客户端的梯度并更新缓冲区。
    
    Args:
        local_trainer: 本地训练器
        idxs_users: 客户端索引列表
        data_iters: 数据迭代器列表
        batch: 当前批次索引
        max_batch: 每轮最大批次数量
        initial_weights: 初始模型权重
        local_weights: 客户端权重列表
        global_gradients: 全局梯度列表（输出）
        key_to_buffer: 提示参数到缓冲区的映射
        use_hcse: 是否使用 HCSE
        epoch: 当前轮次
        
    Returns:
        tuple: (cluster_grads, train_acc_sum, train_acc_count)
            - cluster_grads: 聚类梯度列表（HCSE 模式）或 None
            - train_acc_sum: 训练精度累计和
            - train_acc_count: 训练精度计数
    """
    cluster_grads = [None for _ in idxs_users] if use_hcse else None
    train_acc_sum = 0.0
    train_acc_count = 0
    
    for idx in idxs_users:
        # 加载客户端权重
        if epoch == 0:
            local_trainer.model.load_state_dict(initial_weights, strict=False)
        else:
            local_trainer.model.load_state_dict(local_weights[idx], strict=False)
        
        # 前向传播
        loss_summary = local_trainer.train_forward(
            idx=idx,
            train_iter=data_iters[idx],
            current_batch=batch,
            total_batches=max_batch
        )
        
        # 累积训练精度统计
        if loss_summary is not None and 'acc' in loss_summary:
            train_acc_sum += loss_summary['acc']
            train_acc_count += 1
        
        # 提取全局 prompt 梯度
        local_weight = local_trainer.model.state_dict()
        grad_global = local_trainer.model.prompt_learner.global_ctx.grad
        if grad_global is not None:
            global_gradients[idx] = grad_global.data.clone()
        else:
            global_gradients[idx] = torch.zeros_like(
                local_trainer.model.prompt_learner.global_ctx.data
            )
        
        # 将提示参数写入缓冲区
        for key, buffer in key_to_buffer.items():
            if buffer is None or key not in local_weight:
                continue
            buffer[idx] = copy.deepcopy(local_weight[key])
        
        # HCSE：记录 cluster_ctx 梯度
        if use_hcse and 'prompt_learner.cluster_ctx' in local_weight:
            if local_trainer.model.prompt_learner.cluster_ctx.grad is not None:
                cluster_grads[idx] = local_trainer.model.prompt_learner.cluster_ctx.grad.data.clone()
            else:
                cluster_grads[idx] = torch.zeros_like(
                    local_trainer.model.prompt_learner.cluster_ctx.data
                )
    
    return cluster_grads, train_acc_sum, train_acc_count


def aggregate_gradients_with_hcse(cluster_grads, global_gradients, idxs_users, 
                                   local_trainer, args, logger):
    """
    使用 HCSE 方法聚合梯度。
    
    包括：
    1. 按数据量加权聚合全局梯度
    2. 计算梯度相似度矩阵
    3. Top-k 稀疏化、对称化、指数映射
    4. 构建编码树并聚合聚类梯度
    
    Args:
        cluster_grads: 聚类梯度列表
        global_gradients: 全局梯度列表
        idxs_users: 客户端索引列表
        local_trainer: 本地训练器
        args: 命令行参数对象
        logger: 日志记录器
        
    Returns:
        tuple: (avg_global_gradient, cluster_gradients_to_apply)
            - avg_global_gradient: 聚合后的全局梯度
            - cluster_gradients_to_apply: 分配给各客户端的聚类梯度字典
    """
    # 构建加权全局梯度（按各客户端数据量加权）
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
    
    # 基于 HCSE 的梯度聚类与编码树聚合
    cluster_gradients_to_apply = None
    try:
        from hcse.encoding_tree import (
            PartitionTree,
            compute_gradient_similarity_matrix_torch,
            aggregate_gradients_by_encoding_tree,
        )
        
        # 填充缺失的 cluster_grad
        for i in range(len(cluster_grads)):
            if cluster_grads[i] is None:
                if 'prompt_learner.cluster_ctx' in local_trainer.model.state_dict():
                    zshape = local_trainer.model.prompt_learner.cluster_ctx.data.shape
                    device = local_trainer.model.prompt_learner.cluster_ctx.data.device
                    cluster_grads[i] = torch.zeros(zshape, device=device)
                else:
                    cluster_grads[i] = torch.zeros_like(
                        local_trainer.model.prompt_learner.global_ctx.data
                    )
        
        # 计算梯度相似度矩阵
        sim_mat = compute_gradient_similarity_matrix_torch(cluster_grads, normalize=True)
        
        # Top-k 稀疏化 + 对称化 + 指数映射
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
            # 对称化
            sim_proc = torch.maximum(sim_proc, sim_proc.t())
            # 指数映射增强对高相似度边的区分
            sim_proc = torch.exp(sim_proc)
        
        adj_matrix = sim_proc.detach().cpu().numpy()
        # 基于相似度图构建编码树
        tree = PartitionTree(adj_matrix)
        tree.build_encoding_tree(k=4, mode='v2')
        
        # 沿编码树聚合 cluster_ctx 梯度
        aggregated_cluster_grads = aggregate_gradients_by_encoding_tree(
            tree, cluster_grads, adj_matrix
        )
        
        # 将聚合后的梯度分配给各客户端
        cluster_gradients_to_apply = {
            i: aggregated_cluster_grads[i]
            for i in idxs_users
            if aggregated_cluster_grads[i] is not None
        }
        
        # 返回编码树和邻接矩阵用于梯度聚类可视化
        return avg_global_gradient, cluster_gradients_to_apply, tree, adj_matrix
    except Exception as e:
        logger.warning(f"[HCSE] 聚类与聚合出现异常，跳过本步: {e}")
        tree = None
        adj_matrix = None
    
    return avg_global_gradient, cluster_gradients_to_apply, tree, adj_matrix


def visualize_encoding_tree(tree, adj_matrix, output_dir, epoch, args, logger):
    """
    Visualize encoding tree structure and save as PDF.
    
    Args:
        tree: PartitionTree object
        adj_matrix: Adjacency matrix used to build the tree
        output_dir: Output directory path
        epoch: Current epoch number
        args: Command line arguments
        logger: Logger object
        
    Returns:
        bool: True if visualization succeeded, False otherwise
    """
    if tree is None or adj_matrix is None:
        return False
    
    try:
        from hcse.visualization import EncodingTreeVisualizer
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend to avoid plt.show() blocking
        import matplotlib.pyplot as plt
        
        # Create visualizer
        visualizer = EncodingTreeVisualizer(tree, adj_matrix)
        filename_suffix = build_filename_suffix(args, prefix='')
        tree_save_path = output_dir / f'encoding_tree_e{epoch+1}_{filename_suffix}.pdf'
        
        # Create visualization figure and save as PDF
        # Note: visualize_tree_structure will call plt.show(), but it won't display with Agg backend
        fig = visualizer.visualize_tree_structure(
            figsize=(100, 150),
            save_path=str(tree_save_path)  # Save as PDF (format auto-detected by extension)
        )
        if fig is not None:
            # Ensure file is saved (visualize_tree_structure saves internally, but save again for safety)
            if not tree_save_path.exists() or tree_save_path.stat().st_size == 0:
                fig.savefig(tree_save_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close figure to release memory
            logger.info(f"✅ Encoding tree visualization saved: {tree_save_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"⚠️ Encoding tree visualization failed: {e}")
        return False


def collect_gradient_clustering_data(cluster_grads, idxs_users, local_trainer, tree, cfg, epoch, logger=None):
    """
    收集梯度聚类可视化所需的数据
    
    Args:
        cluster_grads: 簇级梯度列表，每个元素是 torch.Tensor（cluster_ctx 的梯度）
        idxs_users: 客户端索引列表
        local_trainer: 本地训练器
        tree: 编码树对象（可能为 None），用于获取社区划分
        cfg: 配置对象
        epoch: 当前轮次（0-indexed）
        logger: 日志记录器（可选），如果为 None 则使用默认的 logger
        
    Returns:
        dict: 包含以下键的字典：
            - 'gradient_vectors': list[numpy.ndarray] - 每个客户端的梯度向量（flatten后）
            - 'client_labels': list[int] - 每个客户端数据中占主导地位的类别ID
            - 'client_classnames': list[str] - 每个客户端数据中占主导地位的类别名称
            - 'community_ids': list[int] - 每个客户端所属的社区ID（-1表示未分配）
            - 'client_ids': list[int] - 客户端ID列表
            
        注意: gradient_vectors 和 client_ids 只包含 cluster_grads[idx] 不为 None 的客户端
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    if not cfg.GRADIENT_CLUSTERING:
        logger.debug("❌ 梯度聚类未启用: cfg.GRADIENT_CLUSTERING=False")
        return None
    
    # 修改：每个 epoch 都收集数据，而不是只在最后一轮
    # 如果需要只在特定轮次收集，可以通过配置控制
    # target_epoch = cfg.OPTIM.ROUND - 1  # 最后一轮（0-indexed）
    # if epoch != target_epoch:
    #     return None
    
    logger.debug(f"🔍 收集梯度聚类数据: epoch={epoch}, cfg.OPTIM.ROUND={cfg.OPTIM.ROUND}")
    
    data = {
        'gradient_vectors': [],
        'client_labels': [],
        'client_classnames': [],
        'community_ids': [],
        'client_ids': []
    }
    
    # 1. 收集梯度向量（flatten）
    for idx in idxs_users:
        if cluster_grads[idx] is not None:
            grad_vec = cluster_grads[idx].detach().cpu().flatten().numpy()
            data['gradient_vectors'].append(grad_vec)
            data['client_ids'].append(idx)
        else:
            continue
    
    # 2. 收集客户端数据类别标签（占主导地位的类别）
    for idx in idxs_users:
        try:
            # 获取客户端的数据集
            dataset_wrapper = local_trainer.fed_train_loader_x_dict[idx].dataset
            # DatasetWrapper 有一个 data_source 属性，包含原始的 Datum 对象列表
            if hasattr(dataset_wrapper, 'data_source'):
                data_source = dataset_wrapper.data_source
            elif hasattr(dataset_wrapper, 'dataset') and hasattr(dataset_wrapper.dataset, 'data_source'):
                data_source = dataset_wrapper.dataset.data_source
            else:
                # 尝试直接访问
                data_source = dataset_wrapper
            
            # 统计类别分布
            from collections import Counter
            labels = [item.label for item in data_source]
            if len(labels) == 0:
                raise ValueError(f"客户端 {idx} 的数据源为空")
            label_counter = Counter(labels)
            # 获取占主导地位的类别
            dominant_label = label_counter.most_common(1)[0][0]
            data['client_labels'].append(dominant_label)
            
            # 获取占主导地位的类别对应的 classname
            # 找到第一个具有 dominant_label 的 item
            dominant_classname = None
            for item in data_source:
                if item.label == dominant_label:
                    dominant_classname = item.classname
                    break
            data['client_classnames'].append(dominant_classname if dominant_classname else "")
        except Exception as e:
            # 如果无法获取，使用 -1 作为默认值
            logger.debug(f"⚠️ 客户端 {idx} 无法获取标签: {e}")
            data['client_labels'].append(-1)
            data['client_classnames'].append("")
    
    # 3. 收集社区ID（从编码树中获取）
    if tree is not None:
        try:
            # 获取编码树的簇划分
            # 下潜到叶节点：每个叶节点的父节点所包含的所有叶节点，组成一个cluster
            clusters = []
            if tree.root_id is not None:
                # 找到所有叶节点（没有子节点或子节点为空的节点）
                leaf_nodes = []
                for node_id, node in tree.tree_node.items():
                    if not node.children or len(node.children) == 0:
                        leaf_nodes.append(node_id)
                
                # 根据叶节点的父节点进行分组
                parent_to_leaves = {}
                for leaf_id in leaf_nodes:
                    leaf_node = tree.tree_node[leaf_id]
                    parent_id = leaf_node.parent
                    if parent_id is not None:
                        if parent_id not in parent_to_leaves:
                            parent_to_leaves[parent_id] = []
                        # 叶节点的 partition 只包含一个原始图节点
                        parent_to_leaves[parent_id].extend(leaf_node.partition)
                
                # 每个父节点下的所有叶节点组成一个cluster
                clusters = [list(leaves) for leaves in parent_to_leaves.values() if leaves]
            
            # 为每个客户端分配社区ID
            client_to_community = {}
            for comm_id, cluster in enumerate(clusters):
                for client_idx in cluster:
                    client_to_community[client_idx] = comm_id
            
            # 按照 idxs_users 的顺序获取社区ID
            for idx in idxs_users:
                comm_id = client_to_community.get(idx, -1)
                data['community_ids'].append(comm_id)
        except Exception as e:
            # 如果无法获取，使用 -1 作为默认值
            data['community_ids'] = [-1] * len(idxs_users)
    else:
        data['community_ids'] = [-1] * len(idxs_users)
    
    # 使用 PrettyTable 输出数据
    # 表格结构：第1列是community_ids（递增），第2列是client_ids，第3列是client_labels，第4列是client_classnames
    logger.debug(f"数据收集完成: community_ids数量={len(data['community_ids'])}, client_ids数量={len(data['client_ids'])}")
    if len(data['client_ids']) > 0:
        # 创建客户端数据列表，每个元素包含 (community_id, client_id, client_label, client_classname)
        client_data_list = []
        
        # 获取 client_ids 对应的索引在原始 idxs_users 中的位置
        for i, client_id in enumerate(data['client_ids']):
            # 找到这个 client_id 在原始 idxs_users 中的位置
            if client_id in idxs_users:
                idx_pos = idxs_users.index(client_id)
                # 获取对应的 community_id
                if idx_pos < len(data['community_ids']):
                    comm_id = data['community_ids'][idx_pos]
                else:
                    comm_id = -1
                
                # 获取对应的 client_label 和 client_classname
                if idx_pos < len(data['client_labels']):
                    client_label = data['client_labels'][idx_pos]
                else:
                    client_label = -1
                
                if idx_pos < len(data['client_classnames']):
                    client_classname = data['client_classnames'][idx_pos]
                else:
                    client_classname = ""
                
                client_data_list.append((comm_id, client_id, client_label, client_classname))
            else:
                # 如果找不到，使用默认值
                client_data_list.append((-1, client_id, -1, ""))
        
        # 按 community_id 排序（第一列，递增）
        client_data_list.sort(key=lambda x: x[0])
        
        # 创建表格
        table = PrettyTable()
        table.field_names = ['community_id', 'client_id', 'client_label', 'client_classname']
        table.align = "l"
        
        # 为每个客户端添加一行
        for comm_id, client_id, client_label, client_classname in client_data_list:
            table.add_row([comm_id, client_id, client_label, client_classname])
        
        # 使用 print 确保表格能直接输出，同时也使用 logger
        print(f"\n梯度聚类数据汇总 (epoch={epoch}):\n{table}")
        logger.info(f"\n梯度聚类数据汇总 (epoch={epoch}):\n{table}")
    else:
        warning_msg = "⚠️ 没有收集到任何数据"
        print(warning_msg)
        logger.warning(warning_msg)
    
    return data


def apply_differential_privacy_noise(avg_global_gradient, cluster_gradients_to_apply, std):
    """
    为聚合后的梯度添加差分隐私高斯噪声。
    
    Args:
        avg_global_gradient: 聚合后的全局梯度
        cluster_gradients_to_apply: 聚类梯度字典（可能为 None）
        std: 噪声标准差
        
    Returns:
        tuple: (noisy_avg_global_gradient, noisy_cluster_gradients_to_apply)
    """
    # 为全局梯度添加噪声
    noise = torch.normal(
        0, std,
        size=avg_global_gradient.shape,
        device=avg_global_gradient.device
    )
    avg_global_gradient += noise
    
    # 为聚类梯度添加噪声
    if cluster_gradients_to_apply is not None:
        for idx in cluster_gradients_to_apply:
            cluster_grad = cluster_gradients_to_apply[idx]
            if cluster_grad is not None:
                cluster_noise = torch.normal(
                    0, std,
                    size=cluster_grad.shape,
                    device=cluster_grad.device
                )
                cluster_gradients_to_apply[idx] = cluster_grad + cluster_noise
    
    return avg_global_gradient, cluster_gradients_to_apply


def update_client_weights(local_trainer, idxs_users, local_weights, initial_weights,
                          key_to_buffer, avg_global_gradient, cluster_gradients_to_apply,
                          get_prompt_default):
    """
    更新所有客户端的权重。
    
    对每个客户端：
    1. 从缓冲区同步提示参数到 local_weights
    2. 加载权重到模型
    3. 执行反向传播更新
    4. 将更新后的提示参数写回缓冲区
    
    Args:
        local_trainer: 本地训练器
        idxs_users: 客户端索引列表
        local_weights: 客户端权重列表
        initial_weights: 初始模型权重
        key_to_buffer: 提示参数到缓冲区的映射
        avg_global_gradient: 聚合后的全局梯度
        cluster_gradients_to_apply: 分配给各客户端的聚类梯度字典
        get_prompt_default: 获取默认提示参数的函数
    """
    for idx in idxs_users:
        # 从缓冲区同步提示参数到 local_weights
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
        
        # 加载权重并执行反向传播
        local_trainer.model.load_state_dict(local_weights[idx], strict=False)
        
        cluster_grad_for_idx = None
        if cluster_gradients_to_apply is not None and idx in cluster_gradients_to_apply:
            cluster_grad_for_idx = cluster_gradients_to_apply[idx]
        
        local_trainer.train_backward(
            avg_global_gradient=avg_global_gradient,
            aggregated_cluster_gradient=cluster_grad_for_idx,
        )
        
        # 将更新后的提示参数写回缓冲区
        local_weight = local_trainer.model.state_dict()
        for key, buffer in key_to_buffer.items():
            if buffer is None or key not in local_weight:
                continue
            copied = copy.deepcopy(local_weight[key])
            buffer[idx] = copied


def format_results_table(results_list, title, client_ids):
    """
    使用 PrettyTable 格式化测试结果为表格字符串。
    
    Args:
        results_list: 测试结果列表，每个元素是一个包含多个指标的元组
        title: 表格标题
        client_ids: 客户端 ID 列表
        
    Returns:
        str: 格式化后的表格字符串
    """
    if not results_list:
        return ""
    
    num_metrics = len(results_list[0])
    metric_names = ['Accuracy', 'Error Rate', 'Macro F1']
    if num_metrics > 3:
        metric_names.extend([f'Metric {i+4}' for i in range(num_metrics - 3)])
    
    table = PrettyTable()
    table.field_names = ['Client'] + metric_names[:num_metrics]
    table.align['Client'] = 'l'
    for name in metric_names[:num_metrics]:
        table.align[name] = 'r'
    
    # 每个客户端一行
    for idx, res in enumerate(results_list):
        client_id = client_ids[idx] if idx < len(client_ids) else idx
        row = [f'Client {client_id}'] + [f'{val:.2f}' for val in res[:num_metrics]]
        table.add_row(row)
    
    # 最后一行为平均值
    avg_values = []
    for i in range(num_metrics):
        avg_val = sum([res[i] for res in results_list]) / len(results_list) if results_list else 0.0
        avg_values.append(avg_val)
    avg_row = ['Average'] + [f'{val:.2f}' for val in avg_values]
    table.add_row(avg_row)
    
    table_str = f"\n{title}\n{table.get_string()}\n"
    return table_str


def run_test_phase(local_trainer, idxs_users, local_weights, key_to_buffer,
                   dirichlet, epoch, max_epoch, args, logger, wandb_run):
    """
    运行测试阶段，评估所有客户端的模型性能。
    
    Args:
        local_trainer: 本地训练器
        idxs_users: 客户端索引列表
        local_weights: 客户端权重列表
        key_to_buffer: 提示参数到缓冲区的映射
        dirichlet: 是否为 Dirichlet 划分场景
        epoch: 当前轮次
        max_epoch: 最大轮次
        args: 命令行参数对象
        logger: 日志记录器
        wandb_run: wandb 运行对象（可能为 None）
        
    Returns:
        tuple: (avg_local_acc, avg_neighbor_acc)
            - avg_local_acc: 平均本地测试精度
            - avg_neighbor_acc: 平均邻居测试精度（可能为 None）
    """
    show_test_details = getattr(args, "test_verbose_log", False)
    test_start_time = time.time()
    
    if show_test_details:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"                    TEST START - Epoch {epoch + 1}/{max_epoch}")
        logger.info("=" * 80)
        logger.info("")
    
    local_trainer.set_model_mode("eval")
    
    results_local = []
    results_neighbor = [] if not dirichlet else None
    
    test_range = tqdm(idxs_users, desc=f"Epoch {epoch + 1}/{max_epoch} [Test]", leave=False)
    for idx in test_range:
        # 从缓冲区同步提示参数
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
        
        # 评估本地性能
        results_local.append(local_trainer.test(idx=idx, split='local'))
        # 评估邻居性能（仅在非 Dirichlet 场景使用）
        if results_neighbor is not None:
            results_neighbor.append(local_trainer.test(idx=idx, split='neighbor'))
        
        # 更新测试进度条
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
    
    # 格式化并输出测试结果
    local_table = format_results_table(
        results_local,
        "================= Local Test Results ==================",
        idxs_users
    )
    if show_test_details:
        logger.info(local_table)
    
    local_acc = [res[0] for res in results_local]
    avg_local_acc = sum(local_acc) / len(local_acc) if local_acc else 0.0
    
    avg_neighbor_acc = None
    if results_neighbor is not None:
        neighbor_table = format_results_table(
            results_neighbor,
            "================= Neighbor Test Results ==================",
            idxs_users
        )
        if show_test_details:
            logger.info(neighbor_table)
        
        neighbor_acc = [res[0] for res in results_neighbor]
        avg_neighbor_acc = sum(neighbor_acc) / len(neighbor_acc) if neighbor_acc else 0.0
    
    test_duration = time.time() - test_start_time
    if show_test_details:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"                    TEST FINISH - Epoch {epoch + 1}/{max_epoch}")
        logger.info(f"                    测试耗时: {test_duration:.2f}s")
        logger.info("=" * 80)
        logger.info("")
    else:
        neighbor_summary = f"{avg_neighbor_acc:.2f}%" if avg_neighbor_acc is not None else "N/A"
        logger.info(
            f"[Epoch {epoch + 1}/{max_epoch}] 测试耗时: {test_duration:.2f}s | "
            f"local_acc: {avg_local_acc:.2f}% | neighbor_acc: {neighbor_summary}"
        )
    
    # 记录到 wandb
    if wandb_run:
        log_payload = {
            "test/epoch": epoch + 1,
            "test/local_acc_avg": avg_local_acc,
            "test/duration_sec": test_duration,
        }
        if results_neighbor is not None:
            log_payload["test/neighbor_acc_avg"] = avg_neighbor_acc
        wandb.log(log_payload, step=epoch + 1)
    
    return avg_local_acc, avg_neighbor_acc


def save_training_results(args, epoch, local_weights, local_acc_list, neighbor_acc_list):
    """
    保存训练结果：检查点和精度曲线。
    
    Args:
        args: 命令行参数对象
        epoch: 当前轮次
        local_weights: 客户端权重列表
        local_acc_list: 本地精度历史列表
        neighbor_acc_list: 邻居精度历史列表
    """
    save_checkpoint(args, epoch, local_weights, local_acc_list, neighbor_acc_list)
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    filename_suffix = build_filename_suffix(args, prefix='acc_')
    pickle.dump(
        [local_acc_list, neighbor_acc_list],
        open(
            os.path.join(output_dir, f'{filename_suffix}.pkl'),
            'wb'
        )
    )


def log_training_summary(local_acc_list, neighbor_acc_list, dirichlet, logger, wandb_run):
    """
    记录训练摘要：最大精度和平均精度。
    
    Args:
        local_acc_list: 本地精度历史列表
        neighbor_acc_list: 邻居精度历史列表
        dirichlet: 是否为 Dirichlet 划分场景
        logger: 日志记录器
        wandb_run: wandb 运行对象（可能为 None）
    """
    # 计算本地精度统计
    if local_acc_list:
        max_local = max(local_acc_list)
        mean_local = np.mean(local_acc_list[-5:]) if len(local_acc_list) >= 5 else np.mean(local_acc_list)
        logger.info(f"maximum test local acc: {max_local:.3f}")
        logger.info(f"mean of local acc: {mean_local:.3f}")
    else:
        max_local = None
        mean_local = None
    
    # 计算邻居精度统计
    if not dirichlet and neighbor_acc_list:
        max_neighbor = max(neighbor_acc_list)
        mean_neighbor = np.mean(neighbor_acc_list[-5:]) if len(neighbor_acc_list) >= 5 else np.mean(neighbor_acc_list)
        logger.info(f"maximum test neighbor acc: {max_neighbor:.3f}")
        logger.info(f"mean of neighbor acc: {mean_neighbor:.3f}")
    else:
        max_neighbor = None
        mean_neighbor = None
    
    # 记录到 wandb
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




def main(args):
    """
    联邦学习主函数。
    
    执行完整的联邦训练流程：
    1. 初始化环境（日志、配置、训练器）
    2. 初始化提示参数缓冲区
    3. 训练循环（多轮通信）
    4. 测试评估
    5. 保存结果和摘要
    
    Args:
        args: 命令行参数对象
    """
    # ====== 初始化日志与配置 ======
    logger = init_logger_from_args(
        args, 
        log_dir=os.path.expanduser('~/code/sepfpl/logs'), 
        log_to_file=True, 
        log_to_console=True
    )
    
    cfg = setup_cfg(args, mode='full')
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    # ====== 判断是否为 Dirichlet 非IID 划分场景 ======
    # 只在 cifar10 / cifar100 中使用 neighbor 评估逻辑
    dirichlet = args.dataset_config_file.split('/')[-1].split('.')[0] in ['cifar10', 'cifar100']
    
    # ====== 确定算法特性开关 ======
    use_hcse_flag = _use_hcse(args)
    use_time_adaptive_flag = _use_time_adaptive(args)
    
    # ====== 初始化联邦训练相关缓存 ======
    global_gradients = [None for _ in range(args.num_users)]
    local_weights = [{} for _ in range(args.num_users)]
    
    # ====== 构建训练器并保存初始权重 ======
    local_trainer = build_trainer(cfg)
    wandb_run = init_wandb_run(args, cfg, logger)
    if wandb_run:
        wandb.watch(local_trainer.model, log='gradients', log_freq=200, log_graph=False)
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())
    
    # ====== 初始化 cluster_ctx 提示（HCSE 模式） ======
    try:
        if use_hcse_flag and 'prompt_learner.cluster_ctx' in initial_weights:
            for i in range(args.num_users):
                if 'prompt_learner.cluster_ctx' not in local_weights[i]:
                    local_weights[i]['prompt_learner.cluster_ctx'] = copy.deepcopy(
                        initial_weights['prompt_learner.cluster_ctx']
                    )
    except Exception as e:
        logger.warning(f"[Init] cluster_ctx 初始化失败（可忽略，后续训练会自动写入）：{e}")
    
    # ====== 训练过程统计量 ======
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    local_acc_list, neighbor_acc_list = [], []
    
    # ====== 从检查点恢复（如果启用） ======
    if args.resume == 'True':
        start_epoch, local_weights, local_acc_list, neighbor_acc_list = load_checkpoint(args)
        logger.info(f"Resume from epoch {start_epoch}")
    if start_epoch == max_epoch - 1:
        logger.info("已经训练到最后一轮，无需继续")
        return
    
    # ====== 统计每轮最大 batch 数 ======
    train_loaders = getattr(local_trainer, "fed_train_loader_x_dict", {})
    max_batches_per_epoch = max(len(loader) for loader in train_loaders.values()) if train_loaders else 0
    
    # ====== 初始化提示参数缓冲区映射 ======
    key_to_buffer = initialize_prompt_buffers(args, local_weights, use_hcse_flag)
    
    # ====== 定义获取默认提示参数的函数 ======
    def get_prompt_default(key, idx):
        """
        在某些客户端缺失某个提示参数时，提供一个合理的默认值。
        
        Args:
            key: 提示参数键名
            idx: 客户端索引
            
        Returns:
            torch.Tensor: 默认提示参数
        """
        if key == 'prompt_learner.global_ctx':
            return initial_weights['prompt_learner.global_ctx']
        fallback = local_weights[idx].get(
            'prompt_learner.global_ctx', 
            initial_weights['prompt_learner.global_ctx']
        )
        return initial_weights.get(key, fallback)

    # ==================== 全局通信轮主循环 ====================
    for epoch in range(start_epoch, max_epoch):
        idxs_users = list(range(0, cfg.DATASET.USERS))
        
        # ---------- Epoch 初始化：计时与训练精度统计 ----------
        local_trainer.reset_epoch_timer()
        epoch_start_time = time.time()
        train_acc_sum = 0.0
        train_acc_count = 0
        
        # ---------- 时间自适应隐私分配：根据轮数更新噪声标准差 ----------
        if use_time_adaptive_flag and hasattr(local_trainer, 'update_std_for_round'):
            local_trainer.update_std_for_round(epoch)
        
        # ---------- 计算当前 epoch 的全局噪声标准差（若启用 DP） ----------
        std = None
        if args.noise > 0:
            if cfg.MIA:
                std = local_trainer.std * math.sqrt(2)
            else:
                std = local_trainer.std / cfg.DATASET.USERS
        
        # ---------- 为每个客户端准备数据迭代器 ----------
        data_iters = []
        for idx in idxs_users:
            local_trainer.set_model_mode("train")
            loader = local_trainer.fed_train_loader_x_dict[idx]
            data_iters.append(iter(loader))
        
        # ---------- 批次循环：收集梯度并更新提示参数 ----------
        batch_range = tqdm(
            range(0, max_batches_per_epoch),
            desc=f"Epoch {epoch + 1}/{max_epoch} [Batch]",
            leave=False
        )
        
        for batch in batch_range:
            local_trainer.set_model_mode("train")
            
            # ====== 收集客户端梯度 ======
            cluster_grads, batch_acc_sum, batch_acc_count = collect_client_gradients(
                local_trainer, idxs_users, data_iters, batch, max_batches_per_epoch,
                initial_weights, local_weights, global_gradients,
                key_to_buffer, use_hcse_flag, epoch
            )
            train_acc_sum += batch_acc_sum
            train_acc_count += batch_acc_count
            
            # ====== 聚合梯度 ======
            tree = None
            adj_matrix = None
            if use_hcse_flag:
                avg_global_gradient, cluster_gradients_to_apply, tree, adj_matrix = aggregate_gradients_with_hcse(
                    cluster_grads, global_gradients, idxs_users, local_trainer, args, logger
                )
            else:
                # 无 HCSE 时，直接对 global_gradients 取简单平均
                avg_global_gradient = sum(global_gradients) / cfg.DATASET.USERS
                cluster_gradients_to_apply = None
            
            # ====== 收集梯度聚类数据（每个epoch收集一次，在最后一个batch收集） ======
            if cfg.GRADIENT_CLUSTERING and batch == max_batches_per_epoch - 1:
                # 注意：这里收集的是HCSE聚合之前的原始梯度（cluster_grads）
                # tree 是从 aggregate_gradients_with_hcse 中获取的编码树，用于获取社区ID
                clustering_data = collect_gradient_clustering_data(
                    cluster_grads, idxs_users, local_trainer, tree, cfg, epoch, logger=logger
                )
                if clustering_data is not None:
                    # 保存数据到文件
                    from pathlib import Path
                    output_dir = Path(get_output_dir(args))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    filename_suffix = build_filename_suffix(args, prefix='')
                    save_path = output_dir / f'gc_e{epoch+1}_{filename_suffix}.pkl'
                    with open(save_path, 'wb') as f:
                        pickle.dump(clustering_data, f)
                    logger.info(f"✅ 梯度聚类数据已保存: {save_path}")
                
                # Visualize encoding tree and save as PDF
                # visualize_encoding_tree(tree, adj_matrix, output_dir, epoch, args, logger)
            
            # ====== 应用差分隐私噪声 ======
            if std is not None:
                avg_global_gradient, cluster_gradients_to_apply = apply_differential_privacy_noise(
                    avg_global_gradient, cluster_gradients_to_apply, std
                )
            
            # ====== 更新客户端权重 ======
            update_client_weights(
                local_trainer, idxs_users, local_weights, initial_weights,
                key_to_buffer, avg_global_gradient, cluster_gradients_to_apply,
                get_prompt_default
            )
            
            # ====== 更新批次进度条信息 ======
            if train_acc_count > 0:
                avg_acc = train_acc_sum / train_acc_count
                batch_range.set_postfix({'avg_acc': f'{avg_acc:.2f}%'})
            else:
                batch_range.set_postfix({'avg_acc': 'N/A'})
        
        # ====== 每轮训练结束后的日志与 wandb 记录 ======
        train_stage_duration = time.time() - epoch_start_time
        avg_train_acc = (train_acc_sum / train_acc_count) if train_acc_count > 0 else 0.0
        logger.info(
            f"[Epoch {epoch + 1}/{max_epoch}] 训练耗时: {train_stage_duration:.2f}s | "
            f"平均训练准确率: {avg_train_acc:.2f}"
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
        # ==================== 测试阶段 ====================
        # 测试策略：前2个epoch训练完后执行第一次测试，之后在奇数epoch时测试，或者最后一轮也测试
        should_test = not getattr(args, 'skip_test', False) and ((epoch % 2 == 1) or (epoch == max_epoch - 1))
        
        if should_test:
            avg_local_acc, avg_neighbor_acc = run_test_phase(
                local_trainer, idxs_users, local_weights, key_to_buffer,
                dirichlet, epoch, max_epoch, args, logger, wandb_run
            )
            local_acc_list.append(avg_local_acc)
            if avg_neighbor_acc is not None:
                neighbor_acc_list.append(avg_neighbor_acc)
        
        # ---------- 确保 local_weights 包含最新的权重（从 buffer 同步） ----------
        for idx in idxs_users:
            for key, buffer in key_to_buffer.items():
                if buffer is None or buffer[idx] is None:
                    continue
                local_weights[idx][key] = copy.deepcopy(buffer[idx])
        
        # ---------- 保存检查点 & 精度曲线 ----------
        save_training_results(args, epoch, local_weights, local_acc_list, neighbor_acc_list)
    
    # ==================== 训练结束，记录 summary ====================
    log_training_summary(local_acc_list, neighbor_acc_list, dirichlet, logger, wandb_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ====== 通信与训练基本参数 ======
    parser.add_argument('--round', type=int, default=20,
                        help="全局通信轮数（federated rounds）")
    parser.add_argument('--num-users', type=int, default=10,
                        help="客户端数量")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率（learning rate）')
    parser.add_argument('--train-batch-size', type=int, default=32,
                        help="训练 batch size（仅在 useall=True 时生效）")
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help="测试 batch size")
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子（>0 启用固定随机性）")

    # ====== 矩阵分解 & 差分隐私相关参数 ======
    parser.add_argument(
        '--factorization', type=str, default='dpfpl',
        help='矩阵分解 / 联邦提示学习算法：'
             'promptfl, fedotp, fedpgp, dplora, dpfpl, sepfpl, sepfpl_time_adaptive, sepfpl_hcse'
    )
    parser.add_argument('--rank', type=int, default=8,
                        help='矩阵分解的秩（rank）')
    parser.add_argument('--norm-thresh', type=float, default=10.0,
                        help='梯度裁剪的范数阈值（clipping norm）')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='差分隐私高斯噪声尺度（标准差）')
    parser.add_argument('--rdp-alpha', type=float, default=2.0,
                        help='RDP（Rényi DP）阶数 α')
    parser.add_argument('--rdp-p', type=float, default=0.2,
                        help='sepfpl 中时间自适应隐私预算分配的幂次 p')

    # ====== 数据集相关参数 ======
    # caltech101, oxford_flowers, oxford_pets, food101, dtd 等
    parser.add_argument(
        '--iid', default=False,
        help="是否对上述数据集采用 IID 划分（True：IID；False：非 IID）"
    )
    parser.add_argument(
        '--num-shots', type=int, default=16,
        help="few-shot 设置下的每类样本数（仅在 useall=False 时生效）"
    )
    parser.add_argument(
        '--useall', default=True,
        help="是否使用全部训练样本（True：全量训练；False：few-shot）"
    )
    # cifar10, cifar100 等
    parser.add_argument(
        '--partition', type=str, default='noniid-coarsegroup-iid',
        help='cifar10/cifar100 的数据划分策略：'
             '"homo, noniid-labeluni, noniid-labeldir, noniid-labeldir100, noniid-coarsegroup-iid"'
    )
    parser.add_argument(
        '--beta', type=float, default=0.3,
        help='Dirichlet 分布参数 β，用于非 IID 数据划分'
    )

    # ====== 可学习提示（prompt）相关参数 ======
    parser.add_argument(
        '--n_ctx', type=int, default=16,
        help="文本提示的 context token 数量"
    )
    parser.add_argument(
        '--sepfpl-topk', type=int, default=8,
        help='HCSE 中构建相似度图时的 top-k 邻居数（仅 sepfpl 相关方法使用）'
    )

    # ====== 路径相关参数 ======
    parser.add_argument(
        "--root", type=str, default="/datasets",
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--config-file", type=str,
        default="configs/trainers/DP-FPL/vit_b16.yaml",
        help="训练器 / 模型的配置文件路径"
    )
    parser.add_argument(
        "--dataset-config-file", type=str,
        default="configs/datasets/cifar100.yaml",
        help="数据集配置文件路径"
    )
    parser.add_argument(
        "--resume", type=str, default="False",
        help="是否从检查点恢复训练（'True'/'False'）"
    )
    parser.add_argument(
        '--task-id', type=str, default=None,
        help='任务编号标识，格式建议如 "[1/100]"（用于日志与 wandb 标记）'
    )
    # 可选 CLIP 模型路径
    parser.add_argument(
        '--clip-model-path', type=str, default=None,
        help='本地 CLIP 模型文件路径（可选）'
    )
    parser.add_argument(
        '--clip-cache-dir', type=str, default=None,
        help='CLIP 模型下载 / 缓存目录（可选）'
    )

    # ====== wandb 日志配置 ======
    parser.add_argument(
        '--wandb-group', type=str, default=None,
        help='wandb 中的实验分组名（group），用于组织一组相关实验'
    )

    # ====== 测试控制参数 ======
    parser.add_argument(
        '--skip-test', action='store_true', default=False,
        help='跳过测试阶段（默认进行测试）'
    )

    args = parser.parse_args()
    main(args)

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
    扩展 Dassl 默认配置，注入本项目额外需要的配置项。
    """
    from yacs.config import CfgNode as CN

    # ====== 矩阵分解（Factorization）相关参数 ======
    cfg.FACTORIZATION = args.factorization
    cfg.RANK = args.rank

    # ====== 差分隐私（DP）相关参数 ======
    cfg.NORM_THRESH = args.norm_thresh          # 梯度裁剪阈值
    cfg.NOISE = args.noise                      # 高斯噪声尺度
    # RDP（Rényi Differential Privacy）参数
    cfg.RDP_ALPHA = getattr(args, 'rdp_alpha', 2.0)  # RDP 阶数 α
    cfg.RDP_P = getattr(args, 'rdp_p', 1.1)          # sepfpl 中时间自适应隐私分配的幂次 p

    # ====== 训练器（Trainer）相关配置：DP_FPL ======
    cfg.TRAINER.NAME = 'DP_FPL'
    cfg.TRAINER.DP_FPL = CN()
    cfg.TRAINER.DP_FPL.N_CTX = args.n_ctx  # 文本提示 context 个数
    cfg.TRAINER.DP_FPL.PREC = "fp32"       # 计算精度：可选 fp16, fp32, amp
    cfg.TRAINER.DP_FPL.CLASS_TOKEN_POSITION = "end"  # 类别 token 放置位置：'middle' / 'end' / 'front'

    # ====== 数据集相关配置 ======
    cfg.DATASET.ROOT = args.root               # 数据集根路径
    cfg.DATASET.USERS = args.num_users         # 客户端数量
    cfg.DATASET.IID = args.iid                 # 是否采用 IID 划分
    cfg.DATASET.USEALL = args.useall           # 是否使用全部训练样本（否则为 few-shot）
    cfg.DATASET.NUM_SHOTS = args.num_shots     # few-shot 设置下的每类样本数
    cfg.DATASET.PARTITION = args.partition     # cifar 系列的数据划分策略
    cfg.DATASET.BETA = args.beta               # Dirichlet 划分参数

    # Food101 按类下采样比例（不改变类别集合，仅在类内做采样）
    if hasattr(args, 'food101_sample_ratio'):
        cfg.DATASET.FOOD101_SAMPLE_RATIO = args.food101_sample_ratio
    else:
        cfg.DATASET.FOOD101_SAMPLE_RATIO = 1.0

    # CIFAR-100 按类下采样比例（不改变类别集合，仅在类内做采样）
    if hasattr(args, 'cifar100_sample_ratio'):
        cfg.DATASET.CIFAR100_SAMPLE_RATIO = args.cifar100_sample_ratio
    else:
        cfg.DATASET.CIFAR100_SAMPLE_RATIO = 1.0

    # 一些特定数据集（domainnet, office）使用的 domain 数
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 6 if args.num_users == 6 else 4

    # 训练 batch size：USEALL 时使用指定 train_batch_size，否则与 NUM_SHOTS 一致
    if args.useall:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    else:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.num_shots
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # ====== 优化器相关配置 ======
    cfg.OPTIM.ROUND = args.round   # 全局通信轮次
    cfg.OPTIM.MAX_EPOCH = 1        # 每轮本地训练 epoch 数（通常为 1）
    cfg.OPTIM.LR = args.lr         # 学习率

    # ====== 模型与 CLIP 路径相关配置 ======
    cfg.MODEL.BACKBONE.PRETRAINED = True
    if not hasattr(cfg.MODEL.BACKBONE, 'PATH'):
        cfg.MODEL.BACKBONE.PATH = None
    if not hasattr(cfg.MODEL.BACKBONE, 'CACHE_DIR'):
        cfg.MODEL.BACKBONE.CACHE_DIR = None
    if hasattr(args, 'clip_model_path') and args.clip_model_path:
        cfg.MODEL.BACKBONE.PATH = args.clip_model_path
    if hasattr(args, 'clip_cache_dir') and args.clip_cache_dir:
        cfg.MODEL.BACKBONE.CACHE_DIR = args.clip_cache_dir

    # ====== 随机种子 ======
    cfg.SEED = args.seed


def setup_cfg(args):
    """
    构造并返回最终配置：
    1. 读取 Dassl 默认配置；
    2. 根据命令行参数扩展配置；
    3. 从数据集 / 方法配置文件中 merge 对应配置；
    4. 冻结 cfg 防止后续随意修改。
    """
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1) 数据集配置文件（如 configs/datasets/cifar100.yaml）
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2) 算法 / 模型配置文件（如 configs/trainers/DP-FPL/vit_b16.yaml）
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.freeze()
    return cfg


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
    """
    if not _WANDB_AVAILABLE:
        if _should_enable_wandb(args):
            logger.warning("已请求使用 Weights & Biases，但未安装 wandb 包，已自动禁用。")
        return None
    if not _should_enable_wandb(args):
        return None

    project = 'SepFPL'
    group = args.wandb_group       # 实验分组名称
    mode = 'online'                # 默认为 online 模式
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
        "settings": wandb.Settings(start_method="thread", _disable_stats=True),
    }
    # 去除值为 None 的键，以避免 wandb 报错
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    logger.info(f"[wandb] 正在初始化实验：project={project}, name={run_name}, mode={mode}")
    return wandb.init(**init_kwargs)


def save_checkpoint(args, epoch, local_weights, local_acc, neighbor_acc):
    """
    保存模型检查点（包含每个客户端的权重及精度曲线）。
    """
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_dir = os.path.join(os.getcwd(), f'checkpoints/{dataset}')
    os.makedirs(save_dir, exist_ok=True)
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
    """
    从磁盘加载检查点，若不存在则返回默认初始状态。
    """
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_filename = os.path.join(
        os.getcwd(),
        f'/checkpoints/{dataset}/{args.factorization}_{args.rank}_{args.noise}_{args.seed}_{args.num_users}.pth.tar'
    )
    if not os.path.exists(save_filename):
        # epoch=0，local_weights 为 num_users 个空 dict，acc 为空列表
        return 0, [{} for i in range(args.num_users)], [], []
    checkpoint = torch.load(
        save_filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    epoch = checkpoint["epoch"]
    local_weights = checkpoint["local_weights"]
    local_acc = checkpoint["local_acc"]
    neighbor_acc = checkpoint["neighbor_acc"]
    return epoch, local_weights, local_acc, neighbor_acc


def main(args):
    # ====== 初始化日志与配置 ======
    # 日志会将命令行参数与关键信息打印并写入文件，便于后续复现
    logger = init_logger_from_args(args, log_dir='logs', log_to_file=True, log_to_console=True)

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # ====== 判断是否为 Dirichlet 非IID 划分场景（只在 cifar10 / cifar100 中使用 neighbor 评估逻辑） ======
    dirichlet = args.dataset_config_file.split('/')[-1].split('.')[0] in ['cifar10', 'cifar100']

    # 根据当前 factorization 预先确定 HCSE / 时间自适应开关
    use_hcse_flag = _use_hcse(args)
    use_time_adaptive_flag = _use_time_adaptive(args)

    # ====== 初始化联邦训练相关缓存 ======
    global_gradients = [None for _ in range(args.num_users)]  # 每个客户端的全局 prompt 梯度
    local_weights = [{} for _ in range(args.num_users)]       # 每个客户端当前权重
    local_weights_g = [None for _ in range(args.num_users)]   # global_ctx
    local_weights_l = [None for _ in range(args.num_users)]   # local_ctx
    local_weights_u = [None for _ in range(args.num_users)]   # local_u_ctx
    local_weights_v = [None for _ in range(args.num_users)]   # local_v_ctx

    # ====== 构建训练器并保存初始权重 ======
    local_trainer = build_trainer(cfg)
    wandb_run = init_wandb_run(args, cfg, logger)
    if wandb_run:
        # 监控梯度（不记录计算图，以减小开销）
        wandb.watch(local_trainer.model, log='gradients', log_freq=200, log_graph=False)
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())

    # ====== 初始化 sepfpl / sepfpl_hcse 的 cluster_ctx 提示 ======
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

    # 若设置了 resume，则尝试从检查点恢复
    if args.resume == 'True':
        start_epoch, local_weights, local_acc_list, neighbor_acc_list = load_checkpoint(args)
        logger.info(f"Resume from epoch {start_epoch}")
    if start_epoch == max_epoch - 1:
        # 已经训练到最后一轮，无需继续
        return

    # 预先计算全局噪声标准差（若启用 DP）
    if args.noise > 0:
        std = local_trainer.std / cfg.DATASET.USERS

    # ====== 统计每轮最大 batch 数，用于进度条显示 ======
    train_loaders = getattr(local_trainer, "fed_train_loader_x_dict", {})
    if train_loaders:
        max_batches_per_epoch = max(len(loader) for loader in train_loaders.values())
    else:
        max_batches_per_epoch = 0

    # ====== 建立提示参数到缓冲区的映射，便于统一读写 ======
    key_to_buffer = {'prompt_learner.global_ctx': local_weights_g}

    # 是否使用 HCSE（sepfpl / sepfpl_hcse）决定是否维护 cluster_ctx
    use_hcse_for_buffer = use_hcse_flag
    # 针对不同算法变体，决定是否维护 local_ctx / local_u_ctx / local_v_ctx
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
        """
        在某些客户端缺失某个提示参数时，提供一个合理的默认值：
        - global_ctx：直接使用初始权重；
        - 其他：优先尝试使用该客户端的 global_ctx，否则用初始权重。
        """
        if key == 'prompt_learner.global_ctx':
            return initial_weights['prompt_learner.global_ctx']
        fallback = local_weights[idx].get('prompt_learner.global_ctx', initial_weights['prompt_learner.global_ctx'])
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

        # ---------- 为每个客户端准备数据迭代器 ----------
        data_iters = []
        for idx in idxs_users:
            local_trainer.set_model_mode("train")
            loader = local_trainer.fed_train_loader_x_dict[idx]
            data_iters.append(iter(loader))
        max_batch = max_batches_per_epoch

        # 是否使用 HCSE：直接依据 factorization 判断
        use_hcse = use_hcse_flag

        # ---------- 批次循环：收集梯度并更新提示参数 ----------
        batch_range = tqdm(
            range(0, max_batch),
            desc=f"Epoch {epoch + 1}/{max_epoch} [Batch]",
            leave=False
        )
        for batch in batch_range:
            local_trainer.set_model_mode("train")

            # HCSE 模式下，需要收集 cluster_ctx 的梯度
            cluster_grads = [None for _ in idxs_users] if use_hcse else None

            # ====== 逐客户端本地前向与梯度收集 ======
            for idx in idxs_users:
                if epoch == 0:
                    # 首轮从 initial_weights 开始
                    local_trainer.model.load_state_dict(initial_weights, strict=False)
                else:
                    # 之后各轮从各自的 local_weights 开始
                    local_trainer.model.load_state_dict(local_weights[idx], strict=False)

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

                # 取出当前客户端的权重与全局 prompt 梯度
                local_weight = local_trainer.model.state_dict()
                grad_global = local_trainer.model.prompt_learner.global_ctx.grad
                if grad_global is not None:
                    global_gradients[idx] = grad_global.data.clone()
                else:
                    global_gradients[idx] = torch.zeros_like(
                        local_trainer.model.prompt_learner.global_ctx.data
                    )

                # 将各类 prompt 参数写入 buffer，后续用于聚合 / 回传
                for key, buffer in key_to_buffer.items():
                    if buffer is None or key not in local_weight:
                        continue
                    buffer[idx] = copy.deepcopy(local_weight[key])

                # HCSE：记录 cluster_ctx 对应的梯度
                if use_hcse and 'prompt_learner.cluster_ctx' in local_weight:
                    if local_trainer.model.prompt_learner.cluster_ctx.grad is not None:
                        cluster_grads[idx] = local_trainer.model.prompt_learner.cluster_ctx.grad.data.clone()
                    else:
                        cluster_grads[idx] = torch.zeros_like(
                            local_trainer.model.prompt_learner.cluster_ctx.data
                        )

            # ====== 计算全局梯度（以及可选的聚类梯度） ======
            aggregated_cluster_grads = None
            cluster_gradients_to_apply = None

            if use_hcse:
                # ---------- 构建加权全局梯度（按各客户端数据量加权） ----------
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

                # ---------- 基于 HCSE 的梯度聚类与编码树聚合 ----------
                try:
                    from hcse.encoding_tree import (
                        PartitionTree,
                        compute_gradient_similarity_matrix_torch,
                        aggregate_gradients_by_encoding_tree,
                    )

                    # 若有客户端 cluster_grad 缺失，则填充为零向量
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

                    # ====== Top-k 稀疏化 + 对称化 + 指数映射 ======
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
                    tree.build_encoding_tree(k=2, mode='v2')

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
                except Exception as e:
                    logger.warning(f"[HCSE] 聚类与聚合出现异常，跳过本步: {e}")
            else:
                # 无 HCSE 时，直接对 global_gradients 取简单平均
                avg_global_gradient = sum(global_gradients) / cfg.DATASET.USERS

            # 若启用 DP，则在全局梯度上叠加高斯噪声
            if args.noise > 0:
                noise = torch.normal(
                    0, std,
                    size=avg_global_gradient.shape,
                    device=avg_global_gradient.device
                )
                avg_global_gradient += noise

            # ====== 回传与本地权重同步 ======
            for idx in idxs_users:
                # 对各类 prompt 参数，若 buffer 中无有效值，则用默认值填充
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

                # 加载当前客户端的权重并执行“后向一步”
                local_trainer.model.load_state_dict(local_weights[idx], strict=False)

                cluster_grad_for_idx = None
                if cluster_gradients_to_apply is not None and idx in cluster_gradients_to_apply:
                    cluster_grad_for_idx = cluster_gradients_to_apply[idx]

                local_trainer.train_backward(
                    avg_global_gradient=avg_global_gradient,
                    aggregated_cluster_gradient=cluster_grad_for_idx,
                )

                # 回写更新后的提示参数到 buffer
                local_weight = local_trainer.model.state_dict()
                for key, buffer in key_to_buffer.items():
                    if buffer is None or key not in local_weight:
                        continue
                    copied = copy.deepcopy(local_weight[key])
                    buffer[idx] = copied

            # ====== 更新批次进度条信息（训练精度） ======
            if train_acc_count > 0:
                avg_acc = train_acc_sum / train_acc_count
                batch_range.set_postfix({'avg_acc': f'{avg_acc:.2f}%'})
            else:
                batch_range.set_postfix({'avg_acc': 'N/A'})

        # ====== 每轮训练结束后的日志与 wandb 记录 ======
        train_stage_end = time.time()
        train_stage_duration = train_stage_end - epoch_start_time
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

        # ====== 决定是否在本轮进行测试 ======
        should_test = False
        if max_epoch < 20:
            # 通信轮较少：仅在最后一轮做邻居测试
            should_test = (epoch == max_epoch - 1)
        else:
            # 通信轮较多：只在最后 10% 的轮数中进行测试
            last_epochs = max(1, math.ceil(max_epoch / 10))
            should_test = (epoch >= max_epoch - last_epochs)

        if should_test:
            # ==================== 测试阶段 ====================
            test_start_time = time.time()
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"                    TEST START - Epoch {epoch + 1}/{max_epoch}")
            logger.info("=" * 80)
            logger.info("")
            local_trainer.set_model_mode("eval")

            results_local = []                    # 每个客户端的本地测试结果
            results_neighbor = [] if not dirichlet else None  # 邻居测试结果（Dirichlet 场景下可选）

            test_range = tqdm(idxs_users, desc=f"Epoch {epoch + 1}/{max_epoch} [Test]", leave=False)
            for idx in test_range:
                # 将 buffer 中存储的 prompt 参数同步回各客户端的权重
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

                # local split：评估个性化性能
                results_local.append(local_trainer.test(idx=idx, split='local'))
                # neighbor split：评估泛化到“邻居”数据的能力（仅在非 Dirichlet 场景使用）
                if results_neighbor is not None:
                    results_neighbor.append(local_trainer.test(idx=idx, split='neighbor'))

                # 动态更新测试进度条中的平均精度信息
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

            # ---------- 将测试结果格式化为 PrettyTable 表格 ----------
            def format_results_table(results_list, title, client_ids):
                """使用 PrettyTable 格式化测试结果为表格字符串。"""
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

            # Local 测试结果
            local_table = format_results_table(
                results_local,
                "================= Local Test Results ==================",
                idxs_users
            )
            logger.info(local_table)

            local_acc = [res[0] for res in results_local]
            avg_local_acc = sum(local_acc) / len(local_acc) if local_acc else 0.0
            local_acc_list.append(avg_local_acc)

            # Neighbor 测试结果（若存在）
            if results_neighbor is not None:
                neighbor_table = format_results_table(
                    results_neighbor,
                    "================= Neighbor Test Results ==================",
                    idxs_users
                )
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

            # ---------- 保存检查点 & 精度曲线 ----------
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

    # ==================== 训练结束，记录 summary ====================
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

    # 将关键 summary 写入 wandb
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
    parser.add_argument('--rdp-p', type=float, default=1.05,
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
    parser.add_argument(
        '--food101-sample-ratio', type=float, default=0.1,
        help="Food101 每个类别的采样比例 (0,1]，不改变类别集合，仅对类内样本下采样"
    )
    parser.add_argument(
        '--cifar100-sample-ratio', type=float, default=0.2,
        help="CIFAR-100 每个类别的采样比例 (0,1]，不改变类别集合，仅对类内样本下采样"
    )
    # cifar10, cifar100 等
    parser.add_argument(
        '--partition', type=str, default='noniid-labeldir',
        help='cifar10/cifar100 的数据划分策略：'
             '"homo, noniid-labeluni, noniid-labeldir, noniid-labeldir100"'
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

    args = parser.parse_args()
    main(args)

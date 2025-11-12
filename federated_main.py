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
from utils.logger import init_logger_from_args


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

def save_checkpoint(args, epoch, local_weights, local_acc, neighbor_acc):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_dir = os.path.join(os.getcwd(), f'checkpoints/{dataset}')
    os.makedirs(save_dir, exist_ok=True)  # 创建目录
    save_filename = os.path.join(save_dir, f'{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pth.tar')
    state = {
        "epoch": epoch + 1,
        "local_weights": local_weights,
        "local_acc": local_acc,
        "neighbor_acc": neighbor_acc,
    }
    torch.save(state, save_filename)

def load_checkpoint(args):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_filename = os.path.join(os.getcwd(), f'/checkpoints/{dataset}/{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pth.tar')
    if not os.path.exists(save_filename):
        return 0, [{} for i in range(args.num_users)], [], []
    checkpoint = torch.load(save_filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    epoch = checkpoint["epoch"]
    local_weights = checkpoint["local_weights"]
    local_acc = checkpoint["local_acc"]
    neighbor_acc = checkpoint["neighbor_acc"]
    return epoch, local_weights, local_acc, neighbor_acc


def main(args):
    # 初始化日志记录器
    logger = init_logger_from_args(args, log_dir='logs', log_to_file=True, log_to_console=True)
    
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    dirichlet = False
    if args.dataset_config_file.split('/')[-1].split('.')[0] in ['cifar10', 'cifar100']:
        dirichlet = True

    global_gradients = [None for _ in range(args.num_users)]
    local_weights = [{} for _ in range(args.num_users)]
    local_weights_g = [None for _ in range(args.num_users)]
    local_weights_l = [None for _ in range(args.num_users)]
    local_weights_u = [None for _ in range(args.num_users)]
    local_weights_v = [None for _ in range(args.num_users)]

    local_trainer = build_trainer(cfg)
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())

    # sepfpl：首轮为各客户端预填聚类提示参数，若失败由后续流程覆盖
    try:
        if args.factorization == 'sepfpl' and 'prompt_learner.cluster_ctx' in initial_weights:
            for i in range(args.num_users):
                if 'prompt_learner.cluster_ctx' not in local_weights[i]:
                    local_weights[i]['prompt_learner.cluster_ctx'] = copy.deepcopy(initial_weights['prompt_learner.cluster_ctx'])
    except Exception as e:
        logger.warning(f"[Init] cluster_ctx 初始化失败（可忽略首轮由后续流程写入）：{e}")

    # Training
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
    
    # 统计单轮最大批次数，用于进度条估计
    train_loaders = getattr(local_trainer, "fed_train_loader_x_dict", {})
    if train_loaders:
        max_batches_per_epoch = max(len(loader) for loader in train_loaders.values())
    else:
        max_batches_per_epoch = 0
    # 维护提示参数名称到对应缓冲区的映射，便于统一读写逻辑
    key_to_buffer = {'prompt_learner.global_ctx': local_weights_g}
    if args.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl']:
        key_to_buffer['prompt_learner.local_ctx'] = local_weights_l
    if args.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl']:
        key_to_buffer['prompt_learner.local_u_ctx'] = local_weights_u
        key_to_buffer['prompt_learner.local_v_ctx'] = local_weights_v
    if args.factorization == 'sepfpl':
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
        
        # 重置epoch计时器
        local_trainer.reset_epoch_timer()
        epoch_start_time = time.time()
        train_acc_sum = 0.0
        train_acc_count = 0
        
        # 更新sepfpl的增量式隐私预算分配（如果启用）
        if args.factorization == 'sepfpl' and hasattr(local_trainer, 'update_std_for_round'):
            local_trainer.update_std_for_round(epoch)

        # create data iters
        data_iters = []
        for idx in idxs_users:
            local_trainer.set_model_mode("train")
            loader = local_trainer.fed_train_loader_x_dict[idx]
            data_iters.append(iter(loader))
        max_batch = max_batches_per_epoch

        # loop through batches
        for batch in range(0, max_batch):
            local_trainer.set_model_mode("train")
            # 按客户端收集梯度信息，供全局聚合与聚类阶段使用
            cluster_grads = [None for _ in idxs_users] if args.factorization == 'sepfpl' else None
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
                if args.factorization == 'sepfpl' and 'prompt_learner.cluster_ctx' in local_weight:
                    if local_trainer.model.prompt_learner.cluster_ctx.grad is not None:
                        cluster_grads[idx] = local_trainer.model.prompt_learner.cluster_ctx.grad.data.clone()
                    else:
                        cluster_grads[idx] = torch.zeros_like(local_trainer.model.prompt_learner.cluster_ctx.data)

            # average global gradient
            if args.factorization == 'sepfpl':
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
            else:
                avg_global_gradient = sum(global_gradients) / cfg.DATASET.USERS
            if args.noise > 0 and args.factorization != 'sepfpl':
                noise = torch.normal(0, std, size=avg_global_gradient.shape, device=avg_global_gradient.device)
                avg_global_gradient += noise

            # backward and update
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
                local_trainer.train_backward(avg_global_gradient=avg_global_gradient)

                local_weight = local_trainer.model.state_dict()
                for key, buffer in key_to_buffer.items():
                    if buffer is None or key not in local_weight:
                        continue
                    copied = copy.deepcopy(local_weight[key])
                    buffer[idx] = copied
                    local_weights[idx][key] = copied

            # sepfpl：基于结构熵树对簇梯度进行聚合并同步写回
            if args.factorization == 'sepfpl':
                try:
                    from hcse.encoding_tree import PartitionTree, compute_gradient_similarity_matrix_torch, aggregate_gradients_by_encoding_tree
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
                    for idx in idxs_users:
                        if 'prompt_learner.cluster_ctx' in local_weights[idx]:
                            eta_c = args.sepfpl_lr_c if hasattr(args, 'sepfpl_lr_c') and args.sepfpl_lr_c is not None else cfg.OPTIM.LR
                            updated = local_weights[idx]['prompt_learner.cluster_ctx'] - eta_c * aggregated_cluster_grads[idx]
                            detached = updated.detach().clone()
                            local_weights[idx]['prompt_learner.cluster_ctx'] = detached
                            if key_to_buffer.get('prompt_learner.cluster_ctx') is not None:
                                key_to_buffer['prompt_learner.cluster_ctx'][idx] = detached
                except Exception as e:
                    logger.warning(f"[HCSE] 聚类与聚合出现异常，跳过本步: {e}")

        train_stage_end = time.time()
        train_stage_duration = train_stage_end - epoch_start_time
        avg_train_acc = (train_acc_sum / train_acc_count) if train_acc_count > 0 else 0.0
        logger.info(
            f"[Epoch {epoch + 1}/{max_epoch}] 训练耗时: {train_stage_duration:.2f}s | 平均训练准确率: {avg_train_acc:.2f}"
        )

        # test（保持原有频率与输出）
        should_test = False
        if max_epoch < 20:
            # 如果round总数小于20，只在最后1个epoch测试邻居
            should_test = (epoch == max_epoch - 1)
        else:
            # 当round数 >= 20，仅在最后 x = n/10 个 epoch 进行邻居测试
            last_epochs = max(1, math.ceil(max_epoch / 10))
            should_test = (epoch >= max_epoch - last_epochs)

        if should_test:
            test_start_time = time.time()
            logger.info("------------local test start-------------")
            local_trainer.set_model_mode("eval")
            results_local = []
            results_neighbor = [] if not dirichlet else None
            for idx in idxs_users:
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

            local_acc = [res[0] for res in results_local]
            avg_local_acc = sum(local_acc) / len(local_acc) if local_acc else 0.0
            local_acc_list.append(avg_local_acc)
            logger.info(f"Global test local acc: {avg_local_acc:.2f}")

            if results_neighbor is not None:
                neighbor_acc = [res[0] for res in results_neighbor]
                avg_neighbor_acc = sum(neighbor_acc) / len(neighbor_acc) if neighbor_acc else 0.0
                neighbor_acc_list.append(avg_neighbor_acc)
                logger.info(f"Global test neighbor acc: {avg_neighbor_acc:.2f}")

            logger.info("------------local test finish-------------")
            test_duration = time.time() - test_start_time
            logger.info(f"[Epoch {epoch + 1}/{max_epoch}] 测试耗时: {test_duration:.2f}s")

            # save checkpoint（保持）
            save_checkpoint(args, epoch, local_weights, local_acc_list, neighbor_acc_list)
            dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
            output_dir = os.path.join(os.getcwd(), f'outputs/{dataset_name}')
            os.makedirs(output_dir, exist_ok=True)
            pickle.dump([local_acc_list, neighbor_acc_list],
                        open(os.path.join(output_dir, f'acc_{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pkl'), 'wb'))
    logger.info(f"maximum test local acc: {max(local_acc_list):.3f}")
    logger.info(f"mean of local acc: {np.mean(local_acc_list[-5:]):.3f}")
    if not dirichlet:
        logger.info(f"maximum test neighbor acc: {max(neighbor_acc_list):.3f}")
        logger.info(f"mean of neighbor acc: {np.mean(neighbor_acc_list[-5:]):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=20, help="number of communication round")
    parser.add_argument('--num-users', type=int, default=10, help="number of users")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--train-batch-size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test-batch-size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    # parameters of factorization and differential privacy
    parser.add_argument('--factorization', type=str, default='dpfpl', help='Choose from: promptfl, fedotp, fedpgp, dplora, dpfpl, sepfpl')
    parser.add_argument('--rank', type=int, default=8, help='matrix factorization rank')
    parser.add_argument('--norm-thresh', type=float, default=10.0, help='clipping norm threshold')
    parser.add_argument('--noise', type=float, default=0.0, help='differential privacy noise scale')
    parser.add_argument('--rdp-alpha', type=float, default=2.0, help='RDP (Rényi Differential Privacy) order alpha, default 2.0')
    parser.add_argument('--rdp-p', type=float, default=1.1, help='RDP privacy budget allocation parameter p for sepfpl, default 2.0')

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num-shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=True, help="is useall, True for all training samples, False for few shot learning")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.3, help='The parameter for the dirichlet distribution for data partitioning')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    # sepfpl-specific optional params
    parser.add_argument('--sepfpl-topk', type=int, default=8, help='top-k neighbors for HCSE graph sparsification (sepfpl only)')
    parser.add_argument('--sepfpl-lr-c', type=float, default=0.0001, help='learning rate for cluster prompt updates (defaults to OPTIM.LR)')

    # parameters of path
    parser.add_argument("--root", type=str, default="/datasets", help="path to dataset")
    parser.add_argument("--config-file", type=str, default="configs/trainers/DP-FPL/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar100.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default="False", help="resume training or not")
    parser.add_argument('--gpus', type=str, default=None, help="指定可见显卡，如 '0' 或 '0,1'")
    # Optional CLIP model parameters
    parser.add_argument('--clip-model-path', type=str, default=None, help='path to local CLIP model file (optional)')
    parser.add_argument('--clip-cache-dir', type=str, default=None, help='directory to cache/download CLIP models (optional)')

    args = parser.parse_args()
    # 在任何CUDA调用前设置可见显卡
    if args.gpus is not None and len(str(args.gpus).strip()) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        try:
            # 将默认设备设为当前可见设备的第一个
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        except Exception:
            pass
    main(args)


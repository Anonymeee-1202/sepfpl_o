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

    global_gradients = [{} for i in range(args.num_users)]
    local_weights = [{} for i in range(args.num_users)]
    local_weights_g = [[] for i in range(args.num_users)]
    local_weights_l = [[] for i in range(args.num_users)]
    local_weights_u = [[] for i in range(args.num_users)]
    local_weights_v = [[] for i in range(args.num_users)]

    # ========== 初始化全局、cluster、local三个prompt参数结构 ==========
    cluster_prompt_weights = [{} for i in range(args.num_users)]
    # 其余weights保持local_weights用法

    local_trainer = build_trainer(cfg)
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())

    # sepfpl 模式使用简化采样和裁剪参数

    # ========== 新增：为每个客户端初始化cluster_ctx到local_weights中（仅sepfpl） ==========
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
    
    # 初始化epoch时间跟踪
    epoch_times = []  # 存储每个已完成epoch的用时
    
    for epoch in range(start_epoch, max_epoch): # global communication loop
        idxs_users = list(range(0,cfg.DATASET.USERS))
        
        # 显示已完成epoch的时间统计和剩余时间估算
        if epoch > start_epoch and len(epoch_times) > 0:
            avg_epoch_time = np.mean(epoch_times)
            last_epoch_time = epoch_times[-1]
            remaining_epochs = max_epoch - epoch
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            logger.info(f"------------已完成 {epoch - start_epoch} 个epoch | 平均用时: {avg_epoch_time:.2f}s | 上一个epoch: {last_epoch_time:.2f}s | 预计剩余时间: {estimated_remaining_time:.2f}s ({estimated_remaining_time/60:.1f}分钟)-------------")
        
        logger.info(f"------------local train start epoch: {epoch} -------------")
        # 记录epoch开始时间
        epoch_start_time = time.time()
        # 重置epoch计时器
        local_trainer.reset_epoch_timer()
        
        # 更新sepfpl的增量式隐私预算分配（如果启用）
        if args.factorization == 'sepfpl' and hasattr(local_trainer, 'update_std_for_round'):
            local_trainer.update_std_for_round(epoch)

        # 全采样
        sampled_users = idxs_users

        # create data iters
        data_iters = []
        for idx in idxs_users:
            local_trainer.set_model_mode("train")
            loader = local_trainer.fed_train_loader_x_dict[idx]
            data_iters.append(iter(loader))
        max_batch = len(loader)

        # loop through batches
        for batch in range(0, max_batch):
            local_trainer.set_model_mode("train")
            # 收集每客户端的cluster梯度与global梯度（仅sepfpl & sampled users）
            cluster_grads = [None for _ in idxs_users] if args.factorization == 'sepfpl' else None
            for idx in sampled_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(initial_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights[idx], strict=False)
                local_trainer.train_forward(idx=idx, train_iter=data_iters[idx], current_batch=batch, total_batches=max_batch)

                local_weight = local_trainer.model.state_dict()
                global_gradients[idx] = local_trainer.model.prompt_learner.global_ctx.grad.data
                local_weights_g[idx] = copy.deepcopy(local_weight['prompt_learner.global_ctx'])
                if args.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl']:
                    if 'prompt_learner.local_ctx' in local_weight:
                        local_weights_l[idx] = copy.deepcopy(local_weight['prompt_learner.local_ctx'])
                if args.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl']:
                    if 'prompt_learner.local_u_ctx' in local_weight:
                        local_weights_u[idx] = copy.deepcopy(local_weight['prompt_learner.local_u_ctx'])
                    if 'prompt_learner.local_v_ctx' in local_weight:
                        local_weights_v[idx] = copy.deepcopy(local_weight['prompt_learner.local_v_ctx'])
                if args.factorization == 'sepfpl' and 'prompt_learner.cluster_ctx' in local_weight:
                    if local_trainer.model.prompt_learner.cluster_ctx.grad is not None:
                        cluster_grads[idx] = local_trainer.model.prompt_learner.cluster_ctx.grad.data.clone()
                    else:
                        cluster_grads[idx] = torch.zeros_like(local_trainer.model.prompt_learner.cluster_ctx.data)

            # average global gradient
            if args.factorization == 'sepfpl':
                # 构建逐用户全局梯度（加权平均）
                per_user_global = []
                data_sizes = []
                for i in idxs_users:
                    # 获取各客户端数据规模（使用对应DataLoader的dataset长度）
                    try:
                        ds_len = len(local_trainer.fed_train_loader_x_dict[i].dataset)
                    except Exception:
                        ds_len = 1
                    data_sizes.append(max(1, ds_len))
                    per_user_global.append(global_gradients[i])
                # 归一化权重并加权平均
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
                # 确保为未采样或未初始化的客户端填充上一轮或初始权重
                if not isinstance(local_weights_g[idx], torch.Tensor):
                    local_weights_g[idx] = copy.deepcopy(
                        local_weights[idx].get('prompt_learner.global_ctx', initial_weights['prompt_learner.global_ctx'])
                    )
                local_weights[idx]['prompt_learner.global_ctx'] = local_weights_g[idx]

                if args.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl']:
                    if 'prompt_learner.local_ctx' in local_weights[idx]:
                        if not isinstance(local_weights_l[idx], torch.Tensor):
                            local_weights_l[idx] = copy.deepcopy(
                                local_weights[idx].get('prompt_learner.local_ctx', initial_weights.get('prompt_learner.local_ctx', local_weights[idx]['prompt_learner.global_ctx']))
                            )
                        local_weights[idx]['prompt_learner.local_ctx'] = local_weights_l[idx]

                if args.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl']:
                    if 'prompt_learner.local_u_ctx' in local_weights[idx]:
                        if not isinstance(local_weights_u[idx], torch.Tensor):
                            local_weights_u[idx] = copy.deepcopy(
                                local_weights[idx].get('prompt_learner.local_u_ctx', initial_weights.get('prompt_learner.local_u_ctx', local_weights[idx]['prompt_learner.global_ctx']))
                            )
                        local_weights[idx]['prompt_learner.local_u_ctx'] = local_weights_u[idx]
                    if 'prompt_learner.local_v_ctx' in local_weights[idx]:
                        if not isinstance(local_weights_v[idx], torch.Tensor):
                            local_weights_v[idx] = copy.deepcopy(
                                local_weights[idx].get('prompt_learner.local_v_ctx', initial_weights.get('prompt_learner.local_v_ctx', local_weights[idx]['prompt_learner.global_ctx']))
                            )
                        local_weights[idx]['prompt_learner.local_v_ctx'] = local_weights_v[idx]
                local_trainer.model.load_state_dict(local_weights[idx], strict=False)
                local_trainer.train_backward(avg_global_gradient=avg_global_gradient)

                local_weight = local_trainer.model.state_dict()
                local_weights_g[idx] = copy.deepcopy(local_weight['prompt_learner.global_ctx'])
                if args.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl'] and 'prompt_learner.local_ctx' in local_weight:
                    local_weights_l[idx] = copy.deepcopy(local_weight['prompt_learner.local_ctx'])
                if args.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl']:
                    if 'prompt_learner.local_u_ctx' in local_weight:
                        local_weights_u[idx] = copy.deepcopy(local_weight['prompt_learner.local_u_ctx'])
                    if 'prompt_learner.local_v_ctx' in local_weight:
                        local_weights_v[idx] = copy.deepcopy(local_weight['prompt_learner.local_v_ctx'])
                if args.factorization == 'sepfpl' and 'prompt_learner.cluster_ctx' in local_weight:
                    local_weights[idx]['prompt_learner.cluster_ctx'] = copy.deepcopy(local_weight['prompt_learner.cluster_ctx'])

            # 基于cluster梯度的结构熵聚类与簇内聚合（sepfpl）
            if args.factorization == 'sepfpl':
                try:
                    from hcse.encoding_tree import PartitionTree, compute_gradient_similarity_matrix_torch, aggregate_gradients_by_encoding_tree
                    # 处理缺失的cluster梯度
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
                            local_weights[idx]['prompt_learner.cluster_ctx'] = updated.detach().clone()
                except Exception as e:
                    logger.warning(f"[HCSE] 聚类与聚合出现异常，跳过本步: {e}")

            # sepfpl 路径不再维护RDP预算状态

        # test（保持原有频率与输出）
        should_test = False
        if max_epoch < 20:
            # 如果round总数小于20，只在最后1个epoch测试
            should_test = (epoch == max_epoch - 1)
        elif epoch < 90:
            should_test = (epoch % 10 == 0)
        else:
            should_test = ((epoch - 90) % 2 == 0)

        if should_test:
            logger.info("------------local test start-------------")
            local_trainer.set_model_mode("eval")
            results_local, results_neighbor = [], []
            for idx in idxs_users:
                local_weights[idx]['prompt_learner.global_ctx'] = local_weights_g[idx]
                if args.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl'] and 'prompt_learner.local_ctx' in local_weights[idx]:
                    local_weights[idx]['prompt_learner.local_ctx'] = local_weights_l[idx]
                if args.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl']:
                    if 'prompt_learner.local_u_ctx' in local_weights[idx]:
                        local_weights[idx]['prompt_learner.local_u_ctx'] = local_weights_u[idx]
                    if 'prompt_learner.local_v_ctx' in local_weights[idx]:
                        local_weights[idx]['prompt_learner.local_v_ctx'] = local_weights_v[idx]
                # sepfpl时个性化cluster_ctx已在上方写回，这里只需加载
                local_trainer.model.load_state_dict(local_weights[idx], strict=False)

                results_local.append(local_trainer.test(idx=idx, split='local'))
                if not dirichlet:
                    results_neighbor.append(local_trainer.test(idx=idx, split='neighbor'))

            local_acc, neighbor_acc = [], []
            for k in range(len(results_local)):
                local_acc.append(results_local[k][0])
                if not dirichlet:
                    neighbor_acc.append(results_neighbor[k][0])
            local_acc_list.append(sum(local_acc)/len(local_acc))
            logger.info(f"Global test local acc: {sum(local_acc)/len(local_acc)}")
            if not dirichlet:
                neighbor_acc_list.append(sum(neighbor_acc)/len(neighbor_acc))
                logger.info(f"Global test neighbor acc: {sum(neighbor_acc)/len(neighbor_acc)}")
            logger.info("------------local test finish-------------")
            logger.info(f"Epoch: {epoch}/{max_epoch}\tfinished batch : {batch}/{max_batch}")

            # save checkpoint（保持）
            save_checkpoint(args, epoch, local_weights, local_acc_list, neighbor_acc_list)
            dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
            output_dir = os.path.join(os.getcwd(), f'outputs/{dataset_name}')
            os.makedirs(output_dir, exist_ok=True)
            pickle.dump([local_acc_list, neighbor_acc_list],
                        open(os.path.join(output_dir, f'acc_{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pkl'), 'wb'))
        else:
            logger.info(f"Epoch: {epoch}/{max_epoch}\tfinished batch : {batch}/{max_batch} (skip test)")
        
        # 记录epoch用时
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        logger.info(f"------------epoch {epoch} 完成，用时: {epoch_duration:.2f}s-------------")

    logger.info(f"maximum test local acc: {max(local_acc_list)}")
    logger.info(f"mean of local acc: {np.mean(local_acc_list[-5:])}")
    if not dirichlet:
        logger.info(f"maximum test neighbor acc: {max(neighbor_acc_list)}")
        logger.info(f"mean of neighbor acc: {np.mean(neighbor_acc_list[-5:])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=10, help="number of communication round")
    parser.add_argument('--num-users', type=int, default=10, help="number of users")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-batch-size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test-batch-size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    # parameters of factorization and differential privacy
    parser.add_argument('--factorization', type=str, default='dpfpl', help='Choose from: promptfl, fedotp, fedpgp, dplora, dpfpl, sepfpl')
    parser.add_argument('--rank', type=int, default=8, help='matrix factorization rank')
    parser.add_argument('--norm-thresh', type=float, default=10.0, help='clipping norm threshold')
    parser.add_argument('--noise', type=float, default=0.0, help='differential privacy noise scale')

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
    parser.add_argument('--sepfpl-lr-c', type=float, default=None, help='learning rate for cluster prompt updates (defaults to OPTIM.LR)')
    parser.add_argument('--rho-base', type=float, default=1.0, help='base Poisson sampling rate for sepfpl (default 1.0)')
    parser.add_argument('--c-avg', type=float, default=1.0, help='global clipping hyperparam C_avg for compensation noise (sepfpl only)')
    parser.add_argument('--sepfpl-tau', type=int, default=-1, help='transition round tau (default: half of total rounds)')
    parser.add_argument('--sepfpl-rho-conserve', type=float, default=0.5, help='rho in conservation phase (< rho_base)')
    # Note: use '--rho-base' and '--sepfpl-rho-conserve' above; remove unused/duplicate args
    parser.add_argument('--sepfpl-c-avg', type=float, default=None, help='global clipping hyperparam C_avg (defaults to NORM_THRESH)')

    # parameters of path
    parser.add_argument("--root", type=str, default="/datasets", help="path to dataset")
    parser.add_argument("--config-file", type=str, default="configs/trainers/DP-FPL/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar100.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default="False", help="resume training or not")
    parser.add_argument('--gpus', type=str, default=None, help="指定可见显卡，如 '0' 或 '0,1'")

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


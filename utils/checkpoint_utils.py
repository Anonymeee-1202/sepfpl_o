"""
Checkpoint 和文件路径管理工具模块

提供统一的文件名构建、checkpoint保存和加载功能。
"""

import os
import torch


def build_filename_suffix(args, prefix=''):
    """
    构建文件名后缀，对于 sepfpl 相关方法，添加 topk 和 rdp_p 参数。
    
    Args:
        args: 命令行参数对象
        prefix: 文件名前缀（如 'acc_' 或 ''）
    
    Returns:
        文件名后缀字符串（不包含扩展名）
    
    文件名格式：
    - 非 sepfpl 方法: {prefix}{factorization}_{rank}_{noise}_{seed}_{num_users}
    - sepfpl 方法: {prefix}{factorization}_{rank}_{noise}_{seed}_{topk}_{rdp_p}_{num_users}
    """
    # 检查是否是 sepfpl 相关方法
    sepfpl_methods = ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']
    is_sepfpl = hasattr(args, 'factorization') and args.factorization in sepfpl_methods
    
    # 基础参数（固定部分）
    base_parts = [
        args.factorization if hasattr(args, 'factorization') else 'default',
        args.rank if hasattr(args, 'rank') else 8,
        args.noise if hasattr(args, 'noise') else 0.0,
        args.seed if hasattr(args, 'seed') else 1,
    ]
    
    # 如果是 sepfpl 相关方法，添加 topk 和 rdp_p（在 seed 之后，num_users 之前）
    if is_sepfpl:
        sepfpl_topk = getattr(args, 'sepfpl_topk', None)
        rdp_p = getattr(args, 'rdp_p', None)
        
        if sepfpl_topk is not None:
            base_parts.append(sepfpl_topk)  # 直接添加数字，不加前缀
        if rdp_p is not None:
            # 直接使用 rdp_p 的字符串形式，保留原始格式（包含点号）
            base_parts.append(str(rdp_p))
    
    # 最后添加 num_users
    if hasattr(args, 'num_users'):
        base_parts.append(args.num_users)
    
    return f"{prefix}{'_'.join(map(str, base_parts))}"


def get_checkpoint_dir(args):
    """
    获取 checkpoint 目录路径。
    
    Args:
        args: 命令行参数对象
    
    Returns:
        checkpoint 目录路径
    """
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0] if hasattr(args, 'dataset_config_file') else 'default'
    wandb_group = getattr(args, 'wandb_group', None) or 'default'
    return os.path.join(
        os.path.expanduser('~/code/sepfpl/checkpoints'),
        wandb_group,
        dataset
    )


def get_output_dir(args):
    """
    获取输出目录路径。
    
    Args:
        args: 命令行参数对象
    
    Returns:
        输出目录路径
    """
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0] if hasattr(args, 'dataset_config_file') else 'default'
    wandb_group = getattr(args, 'wandb_group', None) or 'default'
    return os.path.join(
        os.path.expanduser('~/code/sepfpl/outputs'),
        wandb_group,
        dataset
    )


def get_checkpoint_path(args, filename_suffix=None):
    """
    获取 checkpoint 文件的完整路径。
    
    Args:
        args: 命令行参数对象
        filename_suffix: 文件名后缀（不含扩展名），如果为 None 则自动生成
    
    Returns:
        checkpoint 文件的完整路径
    """
    checkpoint_dir = get_checkpoint_dir(args)
    if filename_suffix is None:
        filename_suffix = build_filename_suffix(args, prefix='')
    return os.path.join(checkpoint_dir, f'{filename_suffix}.pth.tar')


def save_checkpoint(args, epoch, local_weights, local_acc, neighbor_acc):
    """
    保存模型检查点（包含每个客户端的权重及精度曲线）。
    
    Args:
        args: 命令行参数对象
        epoch: 当前 epoch 编号
        local_weights: 每个客户端的权重列表
        local_acc: 本地测试精度列表
        neighbor_acc: 邻居测试精度列表
    """
    checkpoint_dir = get_checkpoint_dir(args)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename_suffix = build_filename_suffix(args, prefix='')
    save_filename = os.path.join(checkpoint_dir, f'{filename_suffix}.pth.tar')
    
    state = {
        "epoch": epoch + 1,
        "local_weights": local_weights,
        "local_acc": local_acc,
        "neighbor_acc": neighbor_acc,
    }
    torch.save(state, save_filename)


def _find_checkpoint_file(args):
    """
    内部函数：查找 checkpoint 文件路径。
    
    Args:
        args: 命令行参数对象
    
    Returns:
        checkpoint 文件的完整路径，如果找不到则返回 None
    """
    checkpoint_dir = get_checkpoint_dir(args)
    filename_suffix = build_filename_suffix(args, prefix='')
    save_filename = os.path.join(checkpoint_dir, f'{filename_suffix}.pth.tar')
    
    if os.path.exists(save_filename):
        return save_filename
    
    return None


def _load_checkpoint_data(args):
    """
    内部函数：加载 checkpoint 数据。
    
    Args:
        args: 命令行参数对象
    
    Returns:
        checkpoint 字典，如果文件不存在则返回 None
    """
    save_filename = _find_checkpoint_file(args)
    
    if save_filename is None:
        return None
    
    return torch.load(
        save_filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,  # 允许加载自定义类
    )


def load_checkpoint(args):
    """
    从磁盘加载检查点，若不存在则返回默认初始状态。
    
    此函数用于联邦学习训练场景，返回完整的训练状态（epoch、权重、精度曲线等）。
    
    Args:
        args: 命令行参数对象
    
    Returns:
        (epoch, local_weights, local_acc, neighbor_acc) 元组
    """
    checkpoint = _load_checkpoint_data(args)
    
    if checkpoint is None:
        # epoch=0，local_weights 为 num_users 个空 dict，acc 为空列表
        num_users = getattr(args, 'num_users', 10)
        return 0, [{} for i in range(num_users)], [], []
    
    epoch = checkpoint["epoch"]
    local_weights = checkpoint["local_weights"]
    local_acc = checkpoint["local_acc"]
    neighbor_acc = checkpoint["neighbor_acc"]
    return epoch, local_weights, local_acc, neighbor_acc


def load_target_model_weights(args):
    """
    加载目标模型（联邦学习模型）的检查点权重。
    
    此函数主要用于 MIA 攻击场景，只返回模型权重，不返回其他训练状态信息。
    
    Args:
        args: 命令行参数对象
        
    Returns:
        各客户端的模型权重列表
    """
    checkpoint = _load_checkpoint_data(args)
    
    if checkpoint is None:
        # 如果找不到文件，返回空权重
        num_users = getattr(args, 'num_users', 10)
        return [{} for i in range(num_users)]
    
    return checkpoint["local_weights"]


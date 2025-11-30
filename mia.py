"""
成员推理攻击 (Membership Inference Attack, MIA) 模块

本模块实现了针对联邦学习模型的成员推理攻击：
1. 使用 Shadow 模型生成训练数据
2. 训练攻击模型以区分成员和非成员样本
3. 评估攻击模型的成功率

所有训练和测试过程均在 GPU 上执行（如果可用）。
"""

from typing import List, Tuple, Dict, Optional, Callable, Union
import argparse
import os
import glob
import copy
import torch
import numpy as np
import pickle
import random

from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from Dassl.dassl.utils import set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
from Dassl.dassl.utils import read_image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from utils.logger import init_logger_from_args
from tqdm import tqdm
from mia_transforms import build_transform


# ============================================================================
# 常量定义
# ============================================================================

# 图像变换的默认参数（与 federated_main.py 保持一致）
DEFAULT_TRANSFORM_ARGS = dict(
    input_size=(224, 224),
    transform_choices=("random_resized_crop", "random_flip", "normalize"),
    interpolation="bicubic",
    pixel_mean=(0.48145466, 0.4578275, 0.40821073),
    pixel_std=(0.26862954, 0.26130258, 0.27577711),
    crop_padding=4,
    rrcrop_scale=(0.08, 1.0),
    cutout_n=1,
    cutout_len=16,
    gn_mean=0.0,
    gn_std=0.15,
    randaug_n=2,
    randaug_m=10,
    colorjitter_b=0.4,
    colorjitter_c=0.4,
    colorjitter_s=0.4,
    colorjitter_h=0.1,
    rgs_p=0.2,
    gb_k=21,
    gb_p=0.5,
)

# 插值模式映射
INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


# ============================================================================
# 模型定义
# ============================================================================

class AttackModel(nn.Module):
    """
    成员推理攻击模型
    
    输入：目标模型的预测概率向量（posterior probabilities）
    输出：二分类结果（0=非成员, 1=成员）
    
    Args:
        total_classes: 目标模型的类别总数（等于输入特征维度）
    """
    
    def __init__(self, total_classes: int):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(total_classes, 32)
        self.fc2 = nn.Linear(32, 2)  # 二分类：非成员/成员
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 输入特征，形状为 (batch_size, total_classes)
            
        Returns:
            输出概率，形状为 (batch_size, 2)
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# ============================================================================
# 数据集定义
# ============================================================================

class ShadowDataset(Dataset):
    """
    Shadow 数据集，用于训练攻击模型
    
    从 Shadow 模型生成的预测结果中构建训练数据，并进行数据平衡处理。
    
    Args:
        data_samples: 原始数据样本列表，每个样本为 (prediction, membership, label) 元组
            - prediction: 目标模型的预测概率向量
            - membership: 成员标签 (1=成员, 0=非成员)
            - label: 真实类别标签
        target_label: 如果指定，只返回该标签的数据（用于针对特定类别的攻击模型训练）
    """
    
    def __init__(self, data_samples: List[Tuple], target_label: Optional[int] = None):
        label_set = set()
        train_data = []  # 成员样本 (membership=1)
        test_data = []   # 非成员样本 (membership=0)
        
        # 遍历所有样本，按成员身份分类
        for sample in data_samples:
            label_val = sample[2].item() if isinstance(sample[2], torch.Tensor) else sample[2]
            label_set.add(label_val)
            membership = sample[1].item() if isinstance(sample[1], torch.Tensor) else sample[1]
            
            if membership == 1:
                train_data.append(sample)
            else:
                test_data.append(sample)
        
        self.label_set = list(label_set)
        random.shuffle(train_data)
        random.shuffle(test_data)

        # 数据平衡：确保成员和非成员样本数量相等
        # 这对于二分类任务很重要，避免模型偏向多数类
        train_data = train_data[:min(len(train_data), len(test_data))]
        test_data = test_data[:min(len(train_data), len(test_data))]
        self.dataset = train_data + test_data
        
        # 如果指定了 target_label，只保留该标签的数据
        if target_label is not None:
            filtered_dataset = []
            for sample in self.dataset:
                label_val = sample[2].item() if isinstance(sample[2], torch.Tensor) else sample[2]
                if label_val == target_label:
                    filtered_dataset.append(sample)
            self.dataset = filtered_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            (prediction, membership, label) 元组，均为 Tensor 格式
        """
        sample = self.dataset[idx]
        pred, membership, label = sample
        
        # 确保 pred 是 float32 Tensor
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)
        elif pred.dtype != torch.float32:
            pred = pred.float()
        
        # 确保 membership 是 long Tensor
        if not isinstance(membership, torch.Tensor):
            membership = torch.tensor(membership, dtype=torch.long)
        elif membership.dtype != torch.long:
            membership = membership.long()
        
        return pred, membership, label


class ImageDataset(Dataset):
    """
    图像数据集，用于测试阶段加载原始图像
    
    Args:
        dataset: 图像路径和标签列表，每个元素为 (image_path, label)
        input_size: 输入图像尺寸 (height, width)
        interpolation: 插值方法
        pixel_mean: 像素均值（用于归一化）
        pixel_std: 像素标准差（用于归一化）
        normalize_input: 是否进行归一化
        transform: 可选的图像变换函数列表
        return_raw_img: 是否返回原始图像（未应用变换）
    """
    
    def __init__(
        self,
        dataset: List[Tuple[str, int]],
        input_size: Tuple[int, int],
        interpolation: str,
        pixel_mean: Tuple[float, ...],
        pixel_std: Tuple[float, ...],
        normalize_input: bool,
        transform: Optional[Union[List[Callable], Tuple[Callable]]] = None,
        return_raw_img: bool = False,
    ):
        self.dataset = dataset
        transform = transform if transform is not None else []
        if not isinstance(transform, (list, tuple)):
            transform = [transform]
        self.transform = transform
        self.return_raw_img = return_raw_img
        
        # 构建图像预处理流程
        to_tensor = [
            T.Resize(
                input_size,
                interpolation=INTERPOLATION_MODES[interpolation],
            ),
            T.ToTensor(),
        ]
        if normalize_input:
            to_tensor.append(
                T.Normalize(
                    mean=list(pixel_mean),
                    std=pixel_std,
                )
            )
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个图像样本
        
        Returns:
            (image_tensor, label) 元组
        """
        impath, label = self.dataset[idx]
        img_raw = read_image(impath)
        img = img_raw
        
        # 应用数据增强变换
        for tfm in self.transform:
            img = tfm(img)
        
        if self.return_raw_img:
            return (self.to_tensor(img_raw), label)

        # 如果 transform 已经返回 Tensor（例如包含 ToTensor），直接返回；否则再转换
        if isinstance(img, torch.Tensor):
            return img, label
        return self.to_tensor(img), label


# ============================================================================
# 配置管理
# ============================================================================

def extend_cfg(cfg, args, mode: str = 'test'):
    """
    扩展 Dassl 默认配置，注入本项目额外需要的配置项
    
    Args:
        cfg: Dassl 配置对象
        args: 命令行参数
        mode: 'train' 或 'test'，决定需要哪些配置项
    """
    from yacs.config import CfgNode as CN

    # 基础配置（训练和测试都需要）
    cfg.NOISE = args.noise  # 差分隐私噪声尺度
    cfg.DATASET.ROOT = args.root  # 数据集根目录
    cfg.SEED = args.seed  # 随机种子

    # 测试模式需要更多配置
    if mode == 'test':
        # 矩阵分解参数
        cfg.FACTORIZATION = args.factorization
        cfg.RANK = args.rank

        # 差分隐私参数
        cfg.NORM_THRESH = args.norm_thresh

        # DP_FPL Trainer 配置
        cfg.TRAINER.NAME = "DP_FPL"
        cfg.TRAINER.DP_FPL = CN()
        cfg.TRAINER.DP_FPL.N_CTX = args.n_ctx  # 上下文向量数量
        cfg.TRAINER.DP_FPL.PREC = "fp32"  # 计算精度：fp16, fp32, amp
        cfg.TRAINER.DP_FPL.CLASS_TOKEN_POSITION = "end"  # 类别 token 位置

        # 数据集配置
        cfg.DATASET.USERS = args.num_users  # 客户端数量
        cfg.DATASET.IID = args.iid  # 是否 IID 划分
        cfg.DATASET.USEALL = args.useall  # 是否使用全部训练样本
        cfg.DATASET.NUM_SHOTS = args.num_shots  # Few-shot 设置下的每类样本数
        cfg.DATASET.PARTITION = args.partition  # 数据划分策略（CIFAR 系列）
        cfg.DATASET.BETA = args.beta  # Dirichlet 分布参数（CIFAR 系列）
        
        # DataLoader 配置
        cfg.DATALOADER.TRAIN_X.N_DOMAIN = 6 if args.num_users == 6 else 4  # DomainNet/Office
        if args.useall:
            cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
        else:
            cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.num_shots
        cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

        # 优化器配置
        cfg.OPTIM.ROUND = args.round  # 全局通信轮数
        cfg.OPTIM.MAX_EPOCH = 1  # 本地训练轮数
        cfg.OPTIM.LR = args.lr  # 学习率

        cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args, mode: str = 'test'):
    """
    构造并返回最终配置对象
    
    流程：
    1. 读取 Dassl 默认配置
    2. 根据命令行参数扩展配置
    3. 从数据集配置文件中合并配置
    4. 冻结配置防止后续修改
    
    Args:
        args: 命令行参数
        mode: 'train' 或 'test'
        
    Returns:
        冻结后的配置对象
    """
    cfg = get_cfg_default()
    extend_cfg(cfg, args, mode=mode)

    # 合并数据集配置文件
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 测试模式需要模型配置文件
    if mode == 'test' and args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.freeze()
    return cfg


# ============================================================================
# 模型加载函数
# ============================================================================

def load_target(args) -> List[Dict]:
    """
    加载目标模型（联邦学习模型）的检查点
    
    Args:
        args: 命令行参数
        
    Returns:
        各客户端的模型权重列表
    """
    dataset = args.dataset_config_file.split("/")[-1].split(".")[0]
    wandb_group = getattr(args, 'wandb_group', None) or 'default'
    save_filename = os.path.join(
        os.path.expanduser('~/code/sepfpl/checkpoints'),
        wandb_group,
        dataset,
        f'{args.factorization}_{args.rank}_{args.noise}_{args.seed}_{args.num_users}.pth.tar'
    )
    
    if not os.path.exists(save_filename):
        return [{} for i in range(args.num_users)]
    
    checkpoint = torch.load(
        save_filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,  # 允许加载自定义类
    )
    return checkpoint["local_weights"]


def load_attack(args, dataset_name: str, label: int) -> AttackModel:
    """
    加载指定类别的攻击模型检查点
    
    Args:
        args: 命令行参数
        dataset_name: 数据集名称
        label: 类别标签
        
    Returns:
        攻击模型实例
    """
    wandb_group = getattr(args, 'wandb_group', None) or 'default'
    save_filename = os.path.join(
        os.path.expanduser('~/code/sepfpl/checkpoints'),
        wandb_group,
        dataset_name,
        f"mia_{label}_{args.noise}.pth.tar"
    )
    
    attack_model = torch.load(
        save_filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,  # 允许加载自定义类
    )
    return attack_model


# ============================================================================
# 核心训练和测试函数
# ============================================================================

def train_attack_models(args, auto_test: bool = True):
    """
    训练攻击模型的主函数
    
    流程：
    1. 加载 Shadow 数据
    2. 为每个类别训练一个独立的攻击模型
    3. 保存最佳模型
    4. 可选：自动进行测试
    
    Args:
        args: 命令行参数
        auto_test: 训练完成后是否自动进行测试
    """
    # ====== 初始化 ======
    logger = init_logger_from_args(
        args, 
        log_dir=os.path.expanduser('~/code/sepfpl/logs'), 
        log_to_file=True, 
        log_to_console=True
    )
    
    cfg = setup_cfg(args, mode='train')
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    # 确定计算设备（优先使用 GPU）
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.USE_CUDA) else "cpu")
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True  # 优化卷积操作
        logger.info(f"使用 GPU 设备: {device}")
    else:
        logger.info(f"使用 CPU 设备: {device}")

    # ====== 路径设置 ======
    dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
    wandb_group = getattr(args, 'wandb_group', None) or 'default'
    output_dir = os.path.join(os.path.expanduser('~/code/sepfpl/outputs'), wandb_group, dataset_name)
    checkpoint_dir = os.path.join(os.path.expanduser('~/code/sepfpl/checkpoints'), wandb_group, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ====== 加载 Shadow 数据 ======
    train_data = []
    shadow_files_found = []
    shadow_files_missing = []
    
    # 扫描目录中所有匹配的 shadow 文件
    shadow_pattern = f"shadow_{args.noise}_*.pkl"
    existing_files = glob.glob(os.path.join(output_dir, shadow_pattern))
    
    if existing_files:
        # 从文件名提取 seed 号，按 seed 排序加载
        seed_to_file = {}
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            try:
                seed_str = filename.replace(f"shadow_{args.noise}_", "").replace(".pkl", "")
                seed = int(seed_str)
                seed_to_file[seed] = file_path
            except ValueError:
                continue
        
        # 按 seed 排序加载
        for seed in sorted(seed_to_file.keys()):
            shadow_file = seed_to_file[seed]
            try:
                with open(shadow_file, "rb") as f:
                    train_data += pickle.load(f)
                shadow_files_found.append(shadow_file)
            except Exception as e:
                logger.warning(f"加载 shadow 文件失败: {shadow_file}, 错误: {e}")
                shadow_files_missing.append(shadow_file)
    else:
        # 回退：尝试默认 seed 范围 0-49
        logger.warning(f"未在 {output_dir} 中找到匹配 {shadow_pattern} 的文件，尝试使用默认范围 0-49")
        for seed in range(0, 50):
            shadow_file = os.path.join(output_dir, f"shadow_{args.noise}_{seed}.pkl")
            if os.path.exists(shadow_file):
                try:
                    with open(shadow_file, "rb") as f:
                        train_data += pickle.load(f)
                    shadow_files_found.append(shadow_file)
                except Exception as e:
                    logger.warning(f"加载 shadow 文件失败: {shadow_file}, 错误: {e}")
                    shadow_files_missing.append(shadow_file)
    
    if len(train_data) == 0:
        raise ValueError(f"No shadow data found in {output_dir} (pattern: {shadow_pattern})")
    
    logger.info(f"成功加载 {len(shadow_files_found)} 个 shadow 文件")
    if shadow_files_missing:
        logger.warning(f"有 {len(shadow_files_missing)} 个 shadow 文件加载失败")
    
    # ====== 准备数据集 ======
    total_classes = len(train_data[0][0])  # 从第一个样本获取类别数
    full_dataset = ShadowDataset(train_data)
    num_classes = len(full_dataset.label_set)

    # ====== 训练配置 ======
    criterion = torch.nn.CrossEntropyLoss()
    max_epoch = 200
    
    # 优化参数以提高 GPU 利用率
    train_batch_size = 32  # 可根据 GPU 内存调整
    num_workers = 4  # 数据加载并行进程数
    pin_memory = True if torch.cuda.is_available() else False  # 固定内存，加速 GPU 传输
    
    # 混合精度训练（如果支持）
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("启用混合精度训练 (AMP)")

    # ====== 为每个类别训练攻击模型 ======
    for i in range(num_classes):
        class_label = full_dataset.label_set[i]
        logger.info(f"开始训练类别 {class_label} 的攻击模型")
        
        # 为当前类别创建过滤后的数据集
        class_dataset = ShadowDataset(train_data, target_label=class_label)
        
        if len(class_dataset) == 0:
            logger.warning(f"类别 {class_label} 没有数据，跳过")
            continue
        
        # 创建 DataLoader
        train_loader = DataLoader(
            class_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # 保持 worker 进程活跃
        )
        
        # 初始化模型和优化器
        attack_model = AttackModel(total_classes)
        attack_model = attack_model.to(device)  # 移动到 GPU
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)

        best_so_far = torch.inf
        best_model = None

        # ====== 训练循环 ======
        epoch_range = tqdm(range(0, max_epoch), desc=f"Class {class_label}")
        for epoch in epoch_range:
            attack_model.train()
            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            
            for batch in train_loader:
                # 解包 batch: (prediction, membership, label)
                input, output, label = batch
                
                # 确保数据形状正确并移动到 GPU
                if input.dim() == 1:
                    input = input.unsqueeze(0)  # 添加 batch 维度
                input = input.to(device, non_blocking=True)
                
                if output.dim() == 0:
                    output = output.unsqueeze(0)  # 添加 batch 维度
                output = output.to(device, non_blocking=True)
                
                # Forward pass（使用混合精度）
                if use_amp:
                    # 使用新的 autocast API（避免 FutureWarning）
                    with torch.amp.autocast('cuda', enabled=True):
                        pred = attack_model(input)
                        loss = criterion(pred, output)
                else:
                    pred = attack_model(input)
                    loss = criterion(pred, output)
                
                # 记录预测结果（用于后续分析）
                epoch_pred.extend(torch.max(pred.data, 1)[1].cpu().tolist())
                epoch_true.extend(output.cpu().tolist())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()

            # 保存最佳模型
            if epoch_loss < best_so_far:
                best_so_far = epoch_loss
                # 创建 CPU 副本以便保存（不影响训练中的模型）
                best_model = copy.deepcopy(attack_model).cpu()
                epoch_range.set_postfix({'loss': f'{epoch_loss:.2f}', 'best': f'{best_so_far:.2f}'})
            else:
                epoch_range.set_postfix({'loss': f'{epoch_loss:.2f}', 'best': f'{best_so_far:.2f}'})

        # 保存模型
        save_path = os.path.join(checkpoint_dir, f"mia_{class_label}_{args.noise}.pth.tar")
        if best_model is not None:
            torch.save(best_model, save_path)
            logger.info(f"类别 {class_label} 的攻击模型已保存: {save_path}")
    
    logger.info("所有攻击模型训练完成")
    
    # ====== 自动测试 ======
    if auto_test:
        logger.info("=" * 80)
        logger.info("训练完成，开始测试攻击模型...")
        logger.info("=" * 80)
        test_attack_models(args)


def test_attack_models(args):
    """
    测试攻击模型的主函数
    
    流程：
    1. 加载目标模型（联邦学习模型）
    2. 为每个类别加载对应的攻击模型
    3. 在测试集上评估攻击成功率
    4. 保存平均准确率
    
    Args:
        args: 命令行参数
    """
    # ====== 初始化 ======
    logger = init_logger_from_args(
        args, 
        log_dir=os.path.expanduser('~/code/sepfpl/logs'), 
        log_to_file=True, 
        log_to_console=True
    )
    
    cfg = setup_cfg(args, mode='test')
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # ====== 加载目标模型 ======
    local_trainer = build_trainer(cfg)
    local_weights = load_target(args)
    dataset_name = args.dataset_config_file.split("/")[-1].split(".")[0]
    max_idx = local_trainer.max_idx
    local_trainer.model.load_state_dict(local_weights[max_idx], strict=False)
    local_trainer.set_model_mode("eval")
    
    # 确保目标模型在 GPU 上
    device = local_trainer.device
    logger.info(f"目标模型运行在设备: {device}")
    
    # ====== 路径设置 ======
    wandb_group = getattr(args, 'wandb_group', None) or 'default'
    output_dir = os.path.join(os.path.expanduser('~/code/sepfpl/outputs'), wandb_group, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # ====== 准备测试数据 ======
    in_samples = local_trainer.mia_in   # 训练集成员样本
    out_samples = local_trainer.mia_out # 测试集非成员样本
    
    # 提取所有出现的标签
    label_set = set()
    for sample in in_samples:
        label_set.add(sample.label)
    label_list = list(label_set)

    all_correct = []
    
    # ====== 逐类别测试 ======
    for label in label_list:
        logger.info(f"\n测试类别 {label} 的攻击模型")

        # 构建当前类别的测试数据集
        dataset = []
        for sample in in_samples:
            if sample.label == label:
                dataset.append((sample.impath, 1))  # 1 = 成员
        for sample in out_samples:
            if sample.label == label:
                dataset.append((sample.impath, 0))  # 0 = 非成员
        
        if len(dataset) == 0:
            logger.warning(f"类别 {label} 没有测试数据，跳过")
            continue

        # 构建图像变换
        transforms = build_transform(
            **DEFAULT_TRANSFORM_ARGS,
            dataset_name=dataset_name,
            is_train=False,
        )

        # 创建数据集和 DataLoader
        image_dataset = ImageDataset(
            dataset=dataset,
            input_size=DEFAULT_TRANSFORM_ARGS["input_size"],
            transform=transforms,
            interpolation=DEFAULT_TRANSFORM_ARGS["interpolation"],
            pixel_mean=DEFAULT_TRANSFORM_ARGS["pixel_mean"],
            pixel_std=DEFAULT_TRANSFORM_ARGS["pixel_std"],
            normalize_input="normalize" in DEFAULT_TRANSFORM_ARGS["transform_choices"],
        )

        data_loader = DataLoader(
            image_dataset,
            batch_size=32,  # 增加 batch size 以提高 GPU 利用率
            shuffle=True,
            num_workers=4,  # 并行加载数据
            pin_memory=True if torch.cuda.is_available() else False,  # 固定内存加速传输
        )

        # ====== 加载攻击模型 ======
        try:
            attack_model = load_attack(args, dataset_name, label)
            attack_model = attack_model.to(device)  # 移动到 GPU
            attack_model.eval()  # 设置为评估模式
        except FileNotFoundError:
            logger.warning(f"类别 {label} 的攻击模型未找到，跳过")
            continue

        # ====== 测试循环 ======
        correct = 0
        total = 0
        
        with torch.no_grad():  # 测试时不需要梯度
            for target_in, attack_out in data_loader:
                # 将数据移动到 GPU
                target_in = target_in.to(device, non_blocking=True)
                attack_out = attack_out.to(device, non_blocking=True)
                
                # Step 1: 目标模型推理，获取预测概率
                target_out = local_trainer.model_inference(target_in)
                attack_in = F.softmax(target_out, dim=1)
                
                # Step 2: 攻击模型推理，判断是否为成员
                pred = attack_model(attack_in)
                _, predicted = torch.max(pred.data, 1)
                
                # 统计准确率
                correct += (predicted.cpu() == attack_out.cpu()).sum().item()
                total += target_in.size(0)
        
        success_rate = correct / total if total > 0 else 0.0
        logger.info(f'类别 {label} 攻击成功率: {success_rate:.4f} ({correct}/{total})')
        all_correct.append(success_rate)
    
    # ====== 保存结果 ======
    if all_correct:
        avg_success = sum(all_correct) / len(all_correct)
        logger.info(f'\n平均 MIA 成功率: {avg_success:.4f}')
        
        mia_acc_file = os.path.join(output_dir, f'mia_acc_{args.noise}.pkl')
        with open(mia_acc_file, 'wb') as f:
            pickle.dump(avg_success, f)
        logger.info(f"结果已保存: {mia_acc_file}")
    else:
        logger.warning("没有计算任何准确率")


# ============================================================================
# 主函数
# ============================================================================

def main(args):
    """
    主函数，根据 mode 参数决定是训练还是测试
    
    Args:
        args: 命令行参数
    """
    if args.mode == 'train':
        # 训练模式：训练完成后自动进行测试
        train_attack_models(args, auto_test=True)
    elif args.mode == 'test':
        # 测试模式：仅进行测试（用于单独测试已训练的模型）
        test_attack_models(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Must be 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="成员推理攻击 (MIA) 模块")
    
    # ====== 模式选择 ======
    parser.add_argument(
        "--mode", 
        type=str, 
        default="test", 
        choices=["train", "test"],
        help="运行模式: 'train' 训练攻击模型, 'test' 测试攻击模型"
    )
    
    # ====== 训练模式参数 ======
    parser.add_argument(
        "--noise", 
        type=float, 
        default=0.0, 
        help="差分隐私噪声尺度"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/datasets",
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/caltech101.yaml",
        help="数据集配置文件路径"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1,
        help="随机种子（>0 启用固定随机性）"
    )
    parser.add_argument(
        '--wandb-group', 
        type=str, 
        default=None,
        help='wandb 中的实验分组名（group），用于组织一组相关实验'
    )
    
    # ====== 测试模式额外参数 ======
    parser.add_argument(
        "--round", 
        type=int, 
        default=100, 
        help="全局通信轮数 (仅测试模式)"
    )
    parser.add_argument(
        "--num-users", 
        type=int, 
        default=10, 
        help="客户端数量 (仅测试模式)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        help="学习率 (仅测试模式)"
    )
    parser.add_argument(
        "--train-batch-size", 
        type=int, 
        default=32, 
        help="训练批次大小 (仅测试模式)"
    )
    parser.add_argument(
        "--test-batch-size", 
        type=int, 
        default=100, 
        help="测试批次大小 (仅测试模式)"
    )

    # 矩阵分解和差分隐私参数
    parser.add_argument(
        "--factorization",
        type=str,
        default="dpfpl",
        help="矩阵分解方法: promptfl, fedotp, fedpgp, dplora, dpfpl (仅测试模式)",
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=8, 
        help="矩阵分解的秩 (仅测试模式)"
    )
    parser.add_argument(
        "--norm-thresh", 
        type=float, 
        default=10.0, 
        help="梯度裁剪阈值 (仅测试模式)"
    )

    # 数据集参数
    parser.add_argument(
        "--iid",
        default=False,
        help="是否 IID 划分，控制 caltech101, oxford_flowers, oxford_pets, food101, dtd 的 IID 性 (仅测试模式)",
    )
    parser.add_argument(
        "--num-shots", 
        type=int, 
        default=16, 
        help="Few-shot 设置下的每类样本数 (仅测试模式)"
    )
    parser.add_argument(
        "--useall",
        default=True,
        help="是否使用全部训练样本，True=全部样本, False=few-shot (仅测试模式)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="noniid-labeldir",
        help='CIFAR10/100 的数据划分策略: "homo, noniid-labeluni, noniid-labeldir, noniid-labeldir100" (仅测试模式)',
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Dirichlet 分布参数，用于数据划分 (仅测试模式)",
    )

    # 可学习提示参数
    parser.add_argument(
        "--n_ctx", 
        type=int, 
        default=16, 
        help="文本提示的上下文向量数量 (仅测试模式)"
    )

    # 路径参数
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/trainers/DP-FPL/vit_b16.yaml",
        help="模型配置文件路径 (仅测试模式)",
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default="False", 
        help="是否从检查点恢复训练 (仅测试模式)"
    )
    
    # 任务标识参数
    parser.add_argument(
        '--task-id', 
        type=str, 
        default=None,
        help='任务编号标识，格式建议如 "[1/100]"（用于日志与标识）'
    )

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"检测到 {torch.cuda.device_count()} 个 GPU")
        main(args)
    else:
        print("未检测到 GPU，将使用 CPU 运行")
        main(args)

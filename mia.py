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
import math
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
from utils.config_utils import setup_cfg
from utils.checkpoint_utils import (
    build_filename_suffix,
    get_checkpoint_dir,
    get_output_dir,
    load_target_model_weights,
)
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
    成员推理攻击模型（增强版）
    
    输入：目标模型的预测概率向量（posterior probabilities）
    输出：二分类结果（0=非成员, 1=成员）
    
    增强特性：
    - 更深的网络结构（4层全连接）
    - 批归一化（BatchNorm）加速训练并提高稳定性
    - Dropout防止过拟合
    - 更大的隐藏层维度
    - 层间维度逐步递减的设计
    
    Args:
        total_classes: 目标模型的类别总数（等于输入特征维度）
        hidden_dim: 隐藏层维度，默认为512
        dropout_rate: Dropout比例，默认为0.3
    """
    
    def __init__(self, total_classes: int, hidden_dim: int = 512, dropout_rate: float = 0.3):
        super(AttackModel, self).__init__()
        # 增强的网络结构：四层全连接 + BatchNorm + Dropout
        self.fc1 = nn.Linear(total_classes, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 8)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # 输出层（不使用激活函数，让 CrossEntropyLoss 处理）
        self.fc_out = nn.Linear(hidden_dim // 8, 2)  # 二分类：非成员/成员

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 输入特征，形状为 (batch_size, total_classes)
            
        Returns:
            输出 logits，形状为 (batch_size, 2)
        """
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # 第三层
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # 第四层
        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        # 输出层（不使用激活函数，让 CrossEntropyLoss 处理）
        x = self.fc_out(x)
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
        min_size = min(len(train_data), len(test_data))
        train_data = train_data[:min_size]
        test_data = test_data[:min_size]
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
# 模型加载函数
# ============================================================================


def load_attack(args, dataset_name: str) -> AttackModel:
    """
    加载总的攻击模型检查点（不区分noise）
    
    Args:
        args: 命令行参数
        dataset_name: 数据集名称
        
    Returns:
        攻击模型实例
    """
    checkpoint_dir = get_checkpoint_dir(args)
    save_filename = os.path.join(
        checkpoint_dir,
        "mia_all.pth.tar"
    )
    
    attack_model = torch.load(
        save_filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,  # 允许加载自定义类
    )
    return attack_model


# ============================================================================
# Shadow 数据生成函数
# ============================================================================

def generate_shadow_data(args):
    """
    生成 Shadow 数据用于 MIA 攻击模型训练。
    
    此函数与 federated_main.py 中的训练过程完全解耦，可以独立运行：
    1. 读取保存的 checkpoint 文件
    2. 构建 trainer 和数据加载器
    3. 对每个客户端模型进行推理，收集训练集和测试集的预测结果
    4. 保存为 shadow 数据文件（全量数据，不进行采样）
    
    Args:
        args: 命令行参数，需要包含：
            - dataset_config_file: 数据集配置文件
            - config_file: 模型配置文件
            - factorization, rank, noise, seed, num_users: 用于定位 checkpoint
            - sepfpl_topk, rdp_p: sepfpl 相关方法的参数（可选）
            - wandb_group: 实验组名（可选）
    """
    import torch.nn.functional as F
    
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
    
    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    logger.info("生成全量 Shadow 数据（不进行采样）")
    
    # ====== 构建 trainer（会自动创建数据加载器） ======
    logger.info("正在构建 trainer 和数据加载器...")
    local_trainer = build_trainer(cfg)
    local_trainer.set_model_mode("eval")
    
    # ====== 从 checkpoint 加载模型权重 ======
    logger.info("正在从 checkpoint 加载模型权重...")
    local_weights = load_target_model_weights(args)
    
    # 检查是否成功加载权重
    if not local_weights or all(len(w) == 0 for w in local_weights):
        raise ValueError(
            f"未能从 checkpoint 加载模型权重。请检查以下参数是否正确：\n"
            f"  - factorization: {args.factorization}\n"
            f"  - rank: {args.rank}\n"
            f"  - noise: {args.noise}\n"
            f"  - seed: {args.seed}\n"
            f"  - num_users: {args.num_users}\n"
            f"  - wandb_group: {getattr(args, 'wandb_group', 'default')}"
        )
    
    # ====== 路径设置 ======
    dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    
    # ====== 收集预测结果 ======
    num_users = args.num_users
    shadow_data = []
    
    logger.info(f"开始收集所有 {num_users} 个客户端的 Shadow 数据...")
    
    # 遍历所有客户端（使用进度条）
    client_range = tqdm(range(num_users), desc="生成 Shadow 数据")
    for client_idx in client_range:
        client_range.set_description(f"处理客户端 {client_idx+1}/{num_users}")
        
        # 检查该客户端的权重是否存在
        if not local_weights[client_idx] or len(local_weights[client_idx]) == 0:
            logger.warning(f"  客户端 {client_idx} 的权重为空，跳过")
            continue
        
        # 加载当前客户端的模型权重
        local_trainer.model.load_state_dict(local_weights[client_idx], strict=False)
        local_trainer.set_model_mode("eval")
        
        # 收集当前客户端的训练集预测结果
        if client_idx in local_trainer.fed_train_loader_x_dict:
            train_loader = local_trainer.fed_train_loader_x_dict[client_idx]
            client_train_samples = []  # 临时存储该客户端的所有训练样本
            for batch_idx, batch in enumerate(train_loader):
                images = batch["img"].to(local_trainer.device)
                labels = batch["label"].to(local_trainer.device)
                
                with torch.no_grad():
                    outputs = local_trainer.model_inference(images)
                    predictions = F.softmax(outputs, dim=1)
                
                for pred, label in zip(predictions.cpu(), labels.cpu()):
                    # (prediction, membership=1, label)
                    client_train_samples.append((pred, torch.tensor(1), label))
            
            shadow_data.extend(client_train_samples)
            logger.info(f"  客户端 {client_idx} 训练集样本数: {len(client_train_samples)}")
        else:
            logger.warning(f"  客户端 {client_idx} 的训练集数据加载器不存在，跳过")
        
        # 收集当前客户端的测试集预测结果
        if client_idx in local_trainer.fed_test_local_loader_x_dict:
            test_loader = local_trainer.fed_test_local_loader_x_dict[client_idx]
            client_test_samples = []  # 临时存储该客户端的所有测试样本
            for batch_idx, batch in enumerate(test_loader):
                images = batch["img"].to(local_trainer.device)
                labels = batch["label"].to(local_trainer.device)
                
                with torch.no_grad():
                    outputs = local_trainer.model_inference(images)
                    predictions = F.softmax(outputs, dim=1)
                
                for pred, label in zip(predictions.cpu(), labels.cpu()):
                    # (prediction, membership=0, label)
                    client_test_samples.append((pred, torch.tensor(0), label))
            
            shadow_data.extend(client_test_samples)
            logger.info(f"  客户端 {client_idx} 测试集样本数: {len(client_test_samples)}")
        else:
            logger.warning(f"  客户端 {client_idx} 的测试集数据加载器不存在，跳过")
        
        # 更新进度条信息
        total_samples = len(shadow_data)
        client_range.set_postfix({'总样本数': total_samples})
    
    # ====== 保存 shadow 数据 ======
    shadow_file = os.path.join(output_dir, f"shadow_{args.noise}_{args.seed}.pkl")
    
    with open(shadow_file, 'wb') as f:
        pickle.dump(shadow_data, f)
    
    train_count = sum(1 for _, m, _ in shadow_data if m.item() == 1)
    test_count = sum(1 for _, m, _ in shadow_data if m.item() == 0)
    logger.info(f"✅ Shadow 数据已保存: {shadow_file}")
    logger.info(f"   总样本数: {len(shadow_data)}")
    logger.info(f"   训练集样本: {train_count}")
    logger.info(f"   测试集样本: {test_count}")

# ============================================================================
# 核心训练和测试函数
# ============================================================================

def train_attack_models(args):
    """
    训练攻击模型的主函数
    
    流程：
    1. 加载所有noise的 Shadow 数据
    2. 根据 noise_list 和 shadow_sample_ratio_list 的对应关系，对每个noise的数据按照对应的ratio进行采样
    3. 合并所有数据训练一个总的攻击模型（不区分noise和类别）
    4. 保存最佳模型
    
    Args:
        args: 命令行参数，需要包含：
            - noise_list: 噪声值列表（字符串，逗号分隔），例如 "0.0,0.4,0.2"
            - shadow_sample_ratio_list: 采样比例列表（字符串，逗号分隔），例如 "1.0,0.8,0.6"
    """
    # ====== 初始化 ======
    logger = init_logger_from_args(
        args, 
        log_dir=os.path.expanduser('~/code/sepfpl/logs'), 
        log_to_file=True, 
        log_to_console=True
    )
    
    cfg = setup_cfg(args, mode='basic')
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    # 确定计算设备（优先使用 GPU）
    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.USE_CUDA) else "cpu")
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True  # 优化卷积操作
        # 获取当前使用的 GPU 设备信息
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"使用 GPU 设备: {device}")
        logger.info(f"当前 GPU 索引: {current_device}, GPU 名称: {device_name}")
    else:
        logger.info(f"使用 CPU 设备: {device}")

    # ====== 路径设置 ======
    dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
    output_dir = get_output_dir(args)
    checkpoint_dir = get_checkpoint_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ====== 加载所有noise的 Shadow 数据 ======
    # 扫描目录中所有 shadow 文件（不区分noise）
    shadow_pattern = "shadow_*.pkl"
    existing_files = glob.glob(os.path.join(output_dir, shadow_pattern))
    
    if not existing_files:
        raise ValueError(f"No shadow data found in {output_dir} (pattern: {shadow_pattern})")
    
    # 按noise分组组织文件
    noise_to_files = {}  # {noise: [(seed, file_path), ...]}
    
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        try:
            # 解析文件名：shadow_{noise}_{seed}_ratio{ratio}.pkl 或 shadow_{noise}_{seed}.pkl
            # 移除前缀 "shadow_" 和后缀 ".pkl"
            name_without_prefix = filename.replace("shadow_", "")
            name_without_suffix = name_without_prefix.replace(".pkl", "")
            
            # 提取noise（第一个下划线前的部分）
            parts = name_without_suffix.split("_")
            if len(parts) < 2:
                continue
            
            noise_str = parts[0]
            try:
                noise = float(noise_str)
            except ValueError:
                continue
            
            # 提取seed（第二个部分，可能在_ratio之前）
            if "_ratio" in name_without_suffix:
                seed_str = parts[1].split("_ratio")[0]
            else:
                seed_str = parts[1]
            
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            
            if noise not in noise_to_files:
                noise_to_files[noise] = []
            noise_to_files[noise].append((seed, file_path))
        except Exception:
            continue
    
    if not noise_to_files:
        raise ValueError(f"未能解析任何 shadow 文件")
    
    logger.info(f"找到 {len(noise_to_files)} 个不同的 noise 值: {sorted(noise_to_files.keys())}")
    
    # ====== 解析noise_list和shadow_sample_ratio_list ======
    noise_list = []
    shadow_sample_ratio_list = []
    
    # 优先使用新的列表参数
    if hasattr(args, 'noise_list') and args.noise_list:
        try:
            noise_list = [float(x.strip()) for x in args.noise_list.split(',')]
        except ValueError as e:
            raise ValueError(f"无法解析 noise-list: {args.noise_list}, 错误: {e}")
    
    if hasattr(args, 'shadow_sample_ratio_list') and args.shadow_sample_ratio_list:
        try:
            shadow_sample_ratio_list = [float(x.strip()) for x in args.shadow_sample_ratio_list.split(',')]
        except ValueError as e:
            raise ValueError(f"无法解析 shadow-sample-ratio-list: {args.shadow_sample_ratio_list}, 错误: {e}")
    
    # 如果没有提供列表，使用所有找到的noise值，默认全部采样（ratio=1.0）
    if not noise_list:
        noise_list = sorted(noise_to_files.keys())
        shadow_sample_ratio_list = [1.0] * len(noise_list)
        logger.info("未提供noise-list和shadow-sample-ratio-list，使用所有noise值，默认全部采样")
    
    # 检查长度是否一致
    if len(noise_list) != len(shadow_sample_ratio_list):
        raise ValueError(
            f"noise_list长度({len(noise_list)})与shadow_sample_ratio_list长度({len(shadow_sample_ratio_list)})不一致"
        )
    
    # 构建noise到ratio的映射
    noise_to_ratio = dict(zip(noise_list, shadow_sample_ratio_list))
    logger.info(f"使用 {len(noise_list)} 个noise值，对应的采样比例: {shadow_sample_ratio_list}")
    
    # ====== 对每个noise的数据分别加载和采样 ======
    train_data = []
    shadow_files_found = []
    shadow_files_missing = []
    
    # 为采样设置随机种子，确保可重复性
    np.random.seed(cfg.SEED if cfg.SEED >= 0 else 42)
    
    for noise in sorted(noise_to_files.keys()):
        # 如果指定了noise_list，只处理列表中的noise
        if noise_list and noise not in noise_list:
            logger.info(f"跳过 noise {noise}（不在noise_list中）")
            continue
        
        noise_files = sorted(noise_to_files[noise], key=lambda x: x[0])  # 按seed排序
        noise_data = []
        
        # 加载该noise的所有数据
        for seed, file_path in noise_files:
            try:
                with open(file_path, "rb") as f:
                    noise_data += pickle.load(f)
                shadow_files_found.append(file_path)
            except Exception as e:
                logger.warning(f"加载 shadow 文件失败: {file_path}, 错误: {e}")
                shadow_files_missing.append(file_path)
        
        if len(noise_data) == 0:
            logger.warning(f"Noise {noise} 没有有效数据，跳过")
            continue
        
        # 获取该noise对应的采样比例
        sample_ratio = noise_to_ratio.get(noise, 1.0)
        
        # 对该noise的数据进行采样（保持比例）
        if sample_ratio < 1.0:
            original_noise_size = len(noise_data)
            num_samples = max(1, int(len(noise_data) * sample_ratio))
            sampled_indices = np.random.choice(
                len(noise_data), 
                size=num_samples, 
                replace=False
            )
            sampled_noise_data = [noise_data[i] for i in sampled_indices]
            logger.info(f"Noise {noise}: 采样 {len(sampled_noise_data)}/{original_noise_size} (采样比例: {sample_ratio:.2%})")
            train_data.extend(sampled_noise_data)
        else:
            logger.info(f"Noise {noise}: 使用全部数据 {len(noise_data)} 个样本")
            train_data.extend(noise_data)
    
    if len(train_data) == 0:
        raise ValueError(f"没有加载到任何有效的 shadow 数据")
    
    logger.info(f"成功加载 {len(shadow_files_found)} 个 shadow 文件")
    if shadow_files_missing:
        logger.warning(f"有 {len(shadow_files_missing)} 个 shadow 文件加载失败")
    logger.info(f"合并后的总数据量: {len(train_data)} 个样本")
    
    # ====== 准备数据集 ======
    total_classes = len(train_data[0][0])  # 从第一个样本获取类别数
    full_dataset = ShadowDataset(train_data)  # 使用所有类别的数据

    # ====== 训练配置 ======
    criterion = torch.nn.CrossEntropyLoss()
    max_epoch = 100
    
    # 优化参数以提高 GPU 利用率
    train_batch_size = 512  # 可根据 GPU 内存调整
    num_workers = 4  # 数据加载并行进程数
    pin_memory = True if torch.cuda.is_available() else False  # 固定内存，加速 GPU 传输
    
    # 混合精度训练（如果支持）
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("启用混合精度训练 (AMP)")

    # ====== 训练一个总的攻击模型（使用所有类别的数据） ======
    logger.info("开始训练总的攻击模型（使用所有类别的数据）")
    
    # 创建 DataLoader（使用所有数据，不进行类别过滤）
    train_loader = DataLoader(
        full_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # 保持 worker 进程活跃
    )
    
    # 初始化模型和优化器
    attack_model = AttackModel(total_classes)
    attack_model = attack_model.to(device)  # 移动到 GPU
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)

    best_so_far = torch.inf
    best_model = None

    # ====== 训练循环 ======
    epoch_range = tqdm(range(0, max_epoch), desc="Training Attack Model")
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

    # 保存模型（不区分noise）
    save_path = os.path.join(checkpoint_dir, "mia_all.pth.tar")
    if best_model is not None:
        torch.save(best_model, save_path)
        logger.info(f"总的攻击模型已保存: {save_path}")
    
    logger.info("攻击模型训练完成")


def test_attack_models(args):
    """
    测试攻击模型的主函数
    
    流程：
    1. 加载目标模型（联邦学习模型）
    2. 加载总的攻击模型
    3. 在测试集上评估攻击成功率
    4. 分别统计每个label的攻击成功率，以及整体的平均攻击成功率
    5. 保存结果
    
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
    
    cfg = setup_cfg(args, mode='full')
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    # ====== 加载目标模型 ======
    local_trainer = build_trainer(cfg)
    local_weights = load_target_model_weights(args)
    dataset_name = args.dataset_config_file.split("/")[-1].split(".")[0]
    max_idx = local_trainer.max_idx
    local_trainer.model.load_state_dict(local_weights[max_idx], strict=False)
    local_trainer.set_model_mode("eval")
    
    # 确保目标模型在 GPU 上
    device = local_trainer.device
    if torch.cuda.is_available() and cfg.USE_CUDA:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"目标模型运行在设备: {device}")
        logger.info(f"当前 GPU 索引: {current_device}, GPU 名称: {device_name}")
    else:
        logger.info(f"目标模型运行在设备: {device}")
    
    # ====== 路径设置 ======
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # ====== 准备测试数据 ======
    in_samples = local_trainer.mia_in   # 训练集成员样本
    out_samples = local_trainer.mia_out # 测试集非成员样本
    
    # 构建所有类别的测试数据集，同时记录label信息
    dataset = []
    label_to_samples = {}  # {label: [(impath, membership), ...]}
    
    for sample in in_samples:
        dataset.append((sample.impath, 1, sample.label))  # (impath, membership=1, label)
        if sample.label not in label_to_samples:
            label_to_samples[sample.label] = []
        label_to_samples[sample.label].append((sample.impath, 1))
    
    for sample in out_samples:
        dataset.append((sample.impath, 0, sample.label))  # (impath, membership=0, label)
        if sample.label not in label_to_samples:
            label_to_samples[sample.label] = []
        label_to_samples[sample.label].append((sample.impath, 0))
    
    if len(dataset) == 0:
        logger.warning("没有测试数据，退出")
        return

    logger.info(f"总测试样本数: {len(dataset)} (成员: {len(in_samples)}, 非成员: {len(out_samples)})")
    logger.info(f"测试类别数: {len(label_to_samples)}")

    # 构建图像变换
    transforms = build_transform(
        **DEFAULT_TRANSFORM_ARGS,
        dataset_name=dataset_name,
        is_train=False,
    )

    # 创建数据集和 DataLoader（需要修改ImageDataset以返回label）
    # 为了获取label信息，我们需要修改dataset格式
    dataset_for_loader = [(impath, membership) for impath, membership, _ in dataset]
    image_dataset = ImageDataset(
        dataset=dataset_for_loader,
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
        shuffle=False,  # 测试时不应该 shuffle，确保结果可复现
        num_workers=4,  # 并行加载数据
        pin_memory=True if torch.cuda.is_available() else False,  # 固定内存加速传输
    )

    # ====== 加载攻击模型 ======
    try:
        attack_model = load_attack(args, dataset_name)
        attack_model = attack_model.to(device)  # 移动到 GPU
        attack_model.eval()  # 设置为评估模式
        logger.info("成功加载总的攻击模型")
    except FileNotFoundError:
        logger.error(f"攻击模型未找到，请先训练模型")
        return

    # ====== 测试循环 ======
    # 用于统计每个label的准确率
    label_correct = {}  # {label: correct_count}
    label_total = {}    # {label: total_count}
    all_labels = []     # 记录每个样本的label（按batch顺序）
    
    # 构建impath到label的映射
    impath_to_label = {}
    for impath, membership, label in dataset:
        impath_to_label[impath] = label
    
    # 获取所有样本的impath（按DataLoader的顺序）
    all_impaths = [impath for impath, _ in dataset_for_loader]
    
    correct = 0
    total = 0
    batch_start_idx = 0
    
    with torch.no_grad():  # 测试时不需要梯度
        for batch_idx, (target_in, attack_out) in enumerate(data_loader):
            batch_size = target_in.size(0)
            
            # 获取当前batch的label
            batch_impaths = all_impaths[batch_start_idx:batch_start_idx + batch_size]
            batch_labels = [impath_to_label[impath] for impath in batch_impaths]
            all_labels.extend(batch_labels)
            
            # 将数据移动到 GPU
            target_in = target_in.to(device, non_blocking=True)
            attack_out = attack_out.to(device, non_blocking=True)
            
            # Step 1: 目标模型推理，获取预测概率
            target_out = local_trainer.model_inference(target_in)
            
            if hasattr(args, 'noise') and args.noise > 0:
                rdp_alpha = getattr(cfg, 'RDP_ALPHA', 2.0)
                rdp_sigma = math.sqrt(rdp_alpha / (2.0 * args.noise))
                sensitivity = 10.0
                std = rdp_sigma * sensitivity
                noise = torch.randn_like(target_out) * std
                target_out = target_out + noise
            
            attack_in = F.softmax(target_out, dim=1)
            
            # Step 2: 攻击模型推理，判断是否为成员
            pred = attack_model(attack_in)
            _, predicted = torch.max(pred.data, 1)
            
            # 统计整体准确率
            batch_correct = (predicted.cpu() == attack_out.cpu()).sum().item()
            correct += batch_correct
            total += batch_size
            
            # 统计每个label的准确率
            predicted_cpu = predicted.cpu().numpy()
            attack_out_cpu = attack_out.cpu().numpy()
            for i in range(batch_size):
                label = batch_labels[i]
                if label not in label_correct:
                    label_correct[label] = 0
                    label_total[label] = 0
                label_total[label] += 1
                if predicted_cpu[i] == attack_out_cpu[i]:
                    label_correct[label] += 1
            
            batch_start_idx += batch_size
    
    # ====== 计算并输出结果 ======
    success_rate = correct / total if total > 0 else 0.0
    logger.info(f'\n总体攻击成功率: {success_rate:.4f} ({correct}/{total})')
    
    # 计算每个label的准确率
    label_accuracies = {}
    for label in sorted(label_total.keys()):
        label_acc = label_correct[label] / label_total[label] if label_total[label] > 0 else 0.0
        label_accuracies[label] = label_acc
        logger.info(f'类别 {label} 攻击成功率: {label_acc:.4f} ({label_correct[label]}/{label_total[label]})')
    
    # ====== 保存结果 ======
    mia_results = {
        'average': success_rate,
        'per_label': label_accuracies,
        'total_samples': total,
        'correct_samples': correct,
        'per_label_samples': {label: label_total[label] for label in label_total},
        'per_label_correct': {label: label_correct[label] for label in label_correct}
    }
    
    mia_acc_file = os.path.join(output_dir, f'mia_acc_{args.noise}.pkl')
    with open(mia_acc_file, 'wb') as f:
        pickle.dump(mia_results, f)
    logger.info(f"\n结果已保存: {mia_acc_file}")
    logger.info(f"  平均准确率: {success_rate:.4f}")
    logger.info(f"  类别数量: {len(label_accuracies)}")


# ============================================================================
# 主函数
# ============================================================================

def main(args):
    """
    主函数，根据 mode 参数决定是训练、测试、分析还是生成 shadow 数据
    
    Args:
        args: 命令行参数
    """
    if args.mode == 'train':
        # 训练模式：训练攻击模型
        train_attack_models(args)
    elif args.mode == 'test':
        # 测试模式：仅进行测试（用于单独测试已训练的模型）
        test_attack_models(args)
    elif args.mode == 'generate_shadow':
        # Shadow 数据生成模式：从 checkpoint 独立生成 shadow 数据
        generate_shadow_data(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Must be 'train', 'test', or 'generate_shadow'.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="成员推理攻击 (MIA) 模块")
    
    # ====== 模式选择 ======
    parser.add_argument(
        "--mode", 
        type=str, 
        default="test", 
        choices=["train", "test", "generate_shadow"],
        help="运行模式: 'train' 训练攻击模型, 'test' 测试攻击模型, 'generate_shadow' 从checkpoint生成shadow数据"
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
    parser.add_argument(
        '--noise-list',
        type=str,
        default=None,
        help='噪声值列表，用逗号分隔，例如 "0.0,0.4,0.2"。用于 train 模式，指定需要使用的noise值'
    )
    parser.add_argument(
        '--shadow-sample-ratio-list',
        type=str,
        default=None,
        help='Shadow 数据采样比例列表，用逗号分隔，例如 "1.0,0.8,0.6"。用于 train 模式，与noise-list一一对应，每个noise按照对应的ratio进行采样'
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
    parser.add_argument(
        "--sepfpl-topk",
        type=int,
        default=None,
        help="SepFPL top-k 参数（用于 sepfpl 相关方法，仅测试模式）"
    )
    parser.add_argument(
        "--rdp-p",
        type=float,
        default=None,
        help="RDP 时间适应幂次参数（用于 sepfpl 相关方法，仅测试模式）"
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
        default="configs/trainers/vit_b16.yaml",
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

    main(args)


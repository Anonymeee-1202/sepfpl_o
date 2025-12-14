"""
配置管理工具模块

提供统一的配置扩展和设置功能，用于联邦学习和MIA攻击任务。
"""

from Dassl.dassl.config import get_cfg_default


def extend_cfg(cfg, args, mode: str = 'basic'):
    """
    扩展 Dassl 默认配置，注入本项目额外需要的配置项。
    
    Args:
        cfg: Dassl 配置对象
        args: 命令行参数对象
        mode: 配置模式，'basic' 仅基础配置，'full' 完整配置（包含模型、trainer等）
    """
    from yacs.config import CfgNode as CN

    # ====== 基础配置（训练和测试都需要） ======
    cfg.NOISE = args.noise  # 差分隐私噪声尺度
    cfg.DATASET.ROOT = args.root  # 数据集根目录
    cfg.SEED = args.seed  # 随机种子
    # MIA 相关配置（可通过配置文件设置，不通过 args）
    if not hasattr(cfg, 'MIA'):
        cfg.MIA = True  # 默认关闭 MIA 模式
    
    # 模型与 CLIP 路径相关配置
    cfg.MODEL.BACKBONE.PRETRAINED = True
    if not hasattr(cfg.MODEL.BACKBONE, 'PATH'):
        cfg.MODEL.BACKBONE.PATH = None
    if not hasattr(cfg.MODEL.BACKBONE, 'CACHE_DIR'):
        cfg.MODEL.BACKBONE.CACHE_DIR = None
    if hasattr(args, 'clip_model_path') and args.clip_model_path:
        cfg.MODEL.BACKBONE.PATH = args.clip_model_path
    if hasattr(args, 'clip_cache_dir') and args.clip_cache_dir:
        cfg.MODEL.BACKBONE.CACHE_DIR = args.clip_cache_dir

    # 完整模式需要更多配置（包括模型、trainer、数据集等）
    if mode == 'full':
        # ====== 矩阵分解（Factorization）相关参数 ======
        cfg.FACTORIZATION = args.factorization
        cfg.RANK = args.rank

        # ====== 差分隐私（DP）相关参数 ======
        cfg.NORM_THRESH = args.norm_thresh  # 梯度裁剪阈值
        # RDP（Rényi Differential Privacy）参数
        cfg.RDP_ALPHA = getattr(args, 'rdp_alpha', 2.0)  # RDP 阶数 α
        cfg.RDP_P = getattr(args, 'rdp_p', 1.1)  # sepfpl 中时间自适应隐私分配的幂次 p

        # ====== 训练器（Trainer）相关配置 ======
        # 根据 factorization 参数动态选择训练器
        factorization_to_trainer = {
            'promptfl': 'PromptFL',
            'fedotp': 'FedOTP',
            'fedpgp': 'FedPGP',
            'dplora': 'DPLoRA',
            'dpfpl': 'DPFPL',
            'sepfpl': 'SepFPL',
            'sepfpl_time_adaptive': 'SepFPLTimeAdaptive',
            'sepfpl_hcse': 'SepFPLHCSE',
        }
        trainer_name = factorization_to_trainer.get(args.factorization, 'SepFPL')
        cfg.TRAINER.NAME = trainer_name
        
        # 根据训练器名称动态创建配置节点
        # 所有 factorization 方法共享相同的配置结构，但使用训练器特定的配置节点
        if not hasattr(cfg.TRAINER, trainer_name):
            cfg.TRAINER[trainer_name] = CN()
        cfg.TRAINER[trainer_name].N_CTX = args.n_ctx  # 文本提示 context 个数
        cfg.TRAINER[trainer_name].PREC = "fp32"  # 计算精度：可选 fp16, fp32, amp
        cfg.TRAINER[trainer_name].CLASS_TOKEN_POSITION = "end"  # 类别 token 放置位置：'middle' / 'end' / 'front'
        
        # 为了向后兼容，同时设置 DP_FPL 配置（所有训练器共享）
        # 这样 base_trainer.py 中的代码可以继续使用 cfg.TRAINER.DP_FPL
        if not hasattr(cfg.TRAINER, 'DP_FPL'):
            cfg.TRAINER.DP_FPL = CN()
        cfg.TRAINER.DP_FPL.N_CTX = args.n_ctx
        cfg.TRAINER.DP_FPL.PREC = "fp32"
        cfg.TRAINER.DP_FPL.CLASS_TOKEN_POSITION = "end"

        # ====== 数据集相关配置 ======
        cfg.DATASET.USERS = args.num_users  # 客户端数量
        cfg.DATASET.IID = args.iid  # 是否采用 IID 划分
        cfg.DATASET.USEALL = args.useall  # 是否使用全部训练样本（否则为 few-shot）
        cfg.DATASET.NUM_SHOTS = args.num_shots  # few-shot 设置下的每类样本数
        cfg.DATASET.PARTITION = args.partition  # cifar 系列的数据划分策略
        cfg.DATASET.BETA = args.beta  # Dirichlet 划分参数

        # 一些特定数据集（domainnet, office）使用的 domain 数
        cfg.DATALOADER.TRAIN_X.N_DOMAIN = 6 if args.num_users == 6 else 4

        # 训练 batch size：USEALL 时使用指定 train_batch_size，否则与 NUM_SHOTS 一致
        if args.useall:
            cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
        else:
            cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.num_shots
        cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

        # ====== 优化器相关配置 ======
        cfg.OPTIM.ROUND = args.round  # 全局通信轮次
        cfg.OPTIM.MAX_EPOCH = 1  # 每轮本地训练 epoch 数（通常为 1）
        cfg.OPTIM.LR = args.lr  # 学习率

        # SepFPL 相关参数
        sepfpl_topk = getattr(args, 'sepfpl_topk', None)
        rdp_p = getattr(args, 'rdp_p', None)
        if sepfpl_topk is not None:
            cfg.SEPFPL_TOPK = sepfpl_topk
        if rdp_p is not None:
            cfg.RDP_P = rdp_p


def setup_cfg(args, mode: str = 'basic'):
    """
    构造并返回最终配置：
    1. 读取 Dassl 默认配置；
    2. 根据命令行参数扩展配置；
    3. 从数据集 / 方法配置文件中 merge 对应配置；
    4. 冻结 cfg 防止后续随意修改。
    
    Args:
        args: 命令行参数对象
        mode: 配置模式，'basic' 仅基础配置，'full' 完整配置（包含模型、trainer等）
    
    Returns:
        冻结后的配置对象
    """
    cfg = get_cfg_default()
    extend_cfg(cfg, args, mode=mode)

    # 1) 数据集配置文件（如 configs/datasets/cifar100.yaml）
    if hasattr(args, 'dataset_config_file') and args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2) 算法 / 模型配置文件（如 configs/trainers/DP-FPL/vit_b16.yaml）
    # 仅在完整模式下需要
    if mode == 'full' and hasattr(args, 'config_file') and args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.freeze()
    return cfg


from yacs.config import CfgNode as CN

###########################
# 配置定义
###########################

_C = CN()

_C.VERSION = 1  # 配置版本号

# 随机种子设置
# 设置为负值：随机化所有内容
# 设置为正值：使用固定种子（便于复现）
_C.SEED = -1
_C.USE_CUDA = True  # 是否使用 CUDA（GPU）
# 是否打印详细信息
# 例如：训练器、数据集和骨干网络信息
_C.VERBOSE = True

###########################
# 输入配置
###########################
_C.INPUT = CN()
# 输入图像尺寸（高度，宽度）
# _C.INPUT.SIZE = (224, 224)  # ImageNet 标准尺寸
_C.INPUT.SIZE = (32, 32)  # CIFAR 标准尺寸
# 图像缩放时的插值模式
_C.INPUT.INTERPOLATION = "bilinear"  # 双线性插值
# 数据增强变换列表
# 可用选项请参考 transforms.py
_C.INPUT.TRANSFORMS = ()
# 如果为 True，tfm_train 和 tfm_test 将为 None（不使用数据增强）
_C.INPUT.NO_TRANSFORM = False
# 图像归一化的均值和标准差
# 默认值（ImageNet）：
# _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# CIFAR 数据集的均值和标准差
_C.INPUT.PIXEL_MEAN = [0.5071, 0.4865, 0.4409]
_C.INPUT.PIXEL_STD = [0.2673, 0.2564, 0.2762]
# 随机裁剪的填充像素数
_C.INPUT.CROP_PADDING = 4
# 随机缩放裁剪的缩放范围（最小比例，最大比例）
_C.INPUT.RRCROP_SCALE = (0.08, 1.0)
# Cutout 数据增强
_C.INPUT.CUTOUT_N = 1  # Cutout 区域数量
_C.INPUT.CUTOUT_LEN = 16  # Cutout 区域长度
# 高斯噪声
_C.INPUT.GN_MEAN = 0.0  # 噪声均值
_C.INPUT.GN_STD = 0.15  # 噪声标准差
# RandomAugment 数据增强
_C.INPUT.RANDAUGMENT_N = 2  # 随机增强操作的数量
_C.INPUT.RANDAUGMENT_M = 10  # 增强强度（magnitude）
# ColorJitter 颜色抖动（亮度、对比度、饱和度、色调）
_C.INPUT.COLORJITTER_B = 0.4  # 亮度抖动范围
_C.INPUT.COLORJITTER_C = 0.4  # 对比度抖动范围
_C.INPUT.COLORJITTER_S = 0.4  # 饱和度抖动范围
_C.INPUT.COLORJITTER_H = 0.1  # 色调抖动范围
# 随机灰度化的概率
_C.INPUT.RGS_P = 0.2
# 高斯模糊
_C.INPUT.GB_P = 0.5  # 应用该操作的概率
_C.INPUT.GB_K = 21  # 卷积核大小（应为奇数）

###########################
# 数据集配置
###########################
_C.DATASET = CN()
# 数据集存储目录
_C.DATASET.ROOT = ""
_C.DATASET.NAME = ""  # 数据集名称
# 源域/目标域名称列表（字符串）
# 某些数据集有预定义的划分，不适用此配置
_C.DATASET.SOURCE_DOMAINS = ()  # 源域列表
_C.DATASET.TARGET_DOMAINS = ()  # 目标域列表
# 标记样本的总数
# 用于半监督学习
_C.DATASET.NUM_LABELED = -1
# 每个类别的图像数量（Few-shot 学习）
_C.DATASET.NUM_SHOTS = 2
# 验证数据的百分比（仅用于 SSL 数据集）
# 设置为 0 表示不使用验证数据
# 使用验证数据进行超参数调优（参考 Oliver et al. 2018）
_C.DATASET.VAL_PERCENT = 0.1
# STL-10 数据集的折数索引（正常范围是 0-9）
# 负数表示不使用
_C.DATASET.STL10_FOLD = -1
# CIFAR-10/100-C 的损坏类型和强度级别
_C.DATASET.CIFAR_C_TYPE = ""  # 损坏类型
_C.DATASET.CIFAR_C_LEVEL = 1  # 损坏强度级别
# 是否使用未标记数据集中的所有数据（例如 FixMatch）
_C.DATASET.ALL_AS_UNLABELED = False

###########################
# 数据加载器配置
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4  # 数据加载的并行工作进程数
# 对每张图像应用 K 次变换（训练时）
_C.DATALOADER.K_TRANSFORMS = 1
# img0 表示未增强的图像张量
# 用于一致性学习
_C.DATALOADER.RETURN_IMG0 = False
# 训练集（标记数据）数据加载器设置
_C.DATALOADER.TRAIN_X = CN()
_C.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"  # 采样器类型
_C.DATALOADER.TRAIN_X.BATCH_SIZE = 32  # 批次大小
# RandomDomainSampler 的参数
# 0 或 -1 表示从所有域中采样
_C.DATALOADER.TRAIN_X.N_DOMAIN = 0
# RandomClassSampler 的参数
# 每个类别的实例数量
_C.DATALOADER.TRAIN_X.N_INS = 16

# 训练集（未标记数据）数据加载器设置
_C.DATALOADER.TRAIN_U = CN()
# 如果设置为 False，train_u 将使用独立的数据加载器参数
_C.DATALOADER.TRAIN_U.SAME_AS_X = True  # 是否与 train_x 使用相同参数
_C.DATALOADER.TRAIN_U.SAMPLER = "RandomSampler"  # 采样器类型
_C.DATALOADER.TRAIN_U.BATCH_SIZE = 32  # 批次大小
_C.DATALOADER.TRAIN_U.N_DOMAIN = 0  # 域数量
_C.DATALOADER.TRAIN_U.N_INS = 16  # 每个类别的实例数量

# 测试集数据加载器设置
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = "SequentialSampler"  # 顺序采样器
_C.DATALOADER.TEST.BATCH_SIZE = 32  # 批次大小

###########################
# 模型配置
###########################
_C.MODEL = CN()
# 模型权重初始化路径
_C.MODEL.INIT_WEIGHTS = ""
_C.MODEL.BACKBONE = CN()  # 骨干网络配置
_C.MODEL.BACKBONE.NAME = ""  # 骨干网络名称
_C.MODEL.BACKBONE.PRETRAINED = True  # 是否使用预训练权重
# 嵌入层定义
_C.MODEL.HEAD = CN()
# 如果为 None，则不构建嵌入层，
# 骨干网络的输出将直接传递给分类器
_C.MODEL.HEAD.NAME = ""  # 头部网络名称
# 隐藏层结构（列表），例如 [512, 512]
# 如果未定义，则不构建嵌入层
_C.MODEL.HEAD.HIDDEN_LAYERS = ()  # 隐藏层结构
_C.MODEL.HEAD.ACTIVATION = "relu"  # 激活函数
_C.MODEL.HEAD.BN = True  # 是否使用批归一化
_C.MODEL.HEAD.DROPOUT = 0.0  # Dropout 比率

###########################
# 优化器配置
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "adam"  # 优化器名称（如 "adam", "sgd", "adamw" 等）
_C.OPTIM.LR = 0.001  # 学习率
_C.OPTIM.WEIGHT_DECAY = 5e-4  # 权重衰减（L2 正则化系数）
_C.OPTIM.MOMENTUM = 0.9  # SGD 动量系数
_C.OPTIM.SGD_DAMPNING = 0  # SGD 阻尼系数
_C.OPTIM.SGD_NESTEROV = False  # 是否使用 Nesterov 动量
_C.OPTIM.RMSPROP_ALPHA = 0.99  # RMSprop 的平滑系数
# 以下参数也适用于其他自适应优化器（如 AdamW）
_C.OPTIM.ADAM_BETA1 = 0.9  # Adam 优化器的 beta1 参数（一阶矩估计的衰减率）
_C.OPTIM.ADAM_BETA2 = 0.999  # Adam 优化器的 beta2 参数（二阶矩估计的衰减率）
# 分层学习率（STAGED_LR）允许不同层使用不同的学习率
# 例如：预训练的基础层可以使用比新分类层更小的学习率
_C.OPTIM.STAGED_LR = False  # 是否启用分层学习率
_C.OPTIM.NEW_LAYERS = ()  # 新层的名称列表（将使用基础学习率）
_C.OPTIM.BASE_LR_MULT = 0.1  # 基础层学习率倍数（相对于新层）
# 学习率调度器
_C.OPTIM.LR_SCHEDULER = "single_step"  # 学习率调度器类型
# -1 或 0 表示步长等于 max_epoch
_C.OPTIM.STEPSIZE = (-1, )  # 学习率衰减的步长（epoch 数）
_C.OPTIM.GAMMA = 1  # 学习率衰减系数
_C.OPTIM.MAX_EPOCH = 1  # 最大训练轮数
# 设置 WARMUP_EPOCH 大于 0 以启用预热训练
_C.OPTIM.WARMUP_EPOCH = -1  # 预热轮数（-1 表示不使用预热）
# 预热类型：线性（linear）或常数（constant）
_C.OPTIM.WARMUP_TYPE = "linear"  # 预热类型
# 当类型为 constant 时的常数学习率
_C.OPTIM.WARMUP_CONS_LR = 1e-5  # 常数预热学习率
# 当类型为 linear 时的最小学习率
_C.OPTIM.WARMUP_MIN_LR = 1e-5  # 线性预热的最小学习率
# 为下一个调度器重新计数 epoch（last_epoch=-1）
# 否则 last_epoch=warmup_epoch
_C.OPTIM.WARMUP_RECOUNT = True  # 是否重新计数 epoch

###########################
# 训练配置
###########################
_C.TRAIN = CN()
# 训练过程中保存模型的频率（以 epoch 为单位）
# 设置为 0 或负值表示仅保存最后一个模型
_C.TRAIN.CHECKPOINT_FREQ = 0  # 检查点保存频率
# 打印训练信息的频率（以 batch 为单位）
_C.TRAIN.PRINT_FREQ = 10  # 打印频率
# 使用 'train_x'、'train_u' 或 'smaller_one' 来计算
# 一个 epoch 中的迭代次数（用于域适应 DA 和半监督学习 SSL）
_C.TRAIN.COUNT_ITER = "train_x"  # 迭代计数方式

###########################
# 测试配置
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = "Classification"  # 评估器类型
_C.TEST.PER_CLASS_RESULT = False  # 是否显示每个类别的结果
# 是否计算混淆矩阵，将保存到 $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False  # 是否计算混淆矩阵
# 如果 NO_TEST=True，则不进行测试
_C.TEST.NO_TEST = False  # 是否跳过测试
# 用于最终评估的数据集（test 或 val）
_C.TEST.SPLIT = "test"  # 测试集划分
# 训练后测试哪个模型（last_step 或 best_val）
# 如果选择 best_val，每个 epoch 都会进行评估（如果验证数据
# 不可用，将使用测试数据）
_C.TEST.FINAL_MODEL = "last_step"  # 最终测试的模型选择

###########################
# 训练器特定配置
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = ""  # 训练器名称

######
# 域适应（Domain Adaptation, DA）
######
# MCD（Maximum Classifier Discrepancy）
_C.TRAINER.MCD = CN()
_C.TRAINER.MCD.N_STEP_F = 4  # 训练分类器 F 的步数
# MME（Minimum Maximum Entropy）
_C.TRAINER.MME = CN()
_C.TRAINER.MME.LMDA = 0.1  # 熵损失的权重
# CDAC（Conditional Adversarial Domain Adaptation with Clustering）
_C.TRAINER.CDAC = CN()
_C.TRAINER.CDAC.CLASS_LR_MULTI = 10  # 分类器学习率倍数
_C.TRAINER.CDAC.RAMPUP_COEF = 30  # 损失权重增加的系数
_C.TRAINER.CDAC.RAMPUP_ITRS = 1000  # 损失权重增加的迭代次数
_C.TRAINER.CDAC.TOPK_MATCH = 5  # Top-K 匹配数量
_C.TRAINER.CDAC.P_THRESH = 0.95  # 伪标签置信度阈值
_C.TRAINER.CDAC.STRONG_TRANSFORMS = ()  # 强数据增强变换列表
# SE（Self-Ensembling）
_C.TRAINER.SE = CN()
_C.TRAINER.SE.EMA_ALPHA = 0.999  # 指数移动平均的平滑系数
_C.TRAINER.SE.CONF_THRE = 0.95  # 置信度阈值
_C.TRAINER.SE.RAMPUP = 300  # 损失权重增加的迭代次数
# M3SDA（Moment Matching for Multi-Source Domain Adaptation）
_C.TRAINER.M3SDA = CN()
_C.TRAINER.M3SDA.LMDA = 0.5  # 矩距离损失的权重
_C.TRAINER.M3SDA.N_STEP_F = 4  # 训练分类器 F 的步数（参考 MCD）
# DAEL（Domain Adaptation with Entropy Minimization and Labeling）
_C.TRAINER.DAEL = CN()
_C.TRAINER.DAEL.WEIGHT_U = 0.5  # 未标记数据损失的权重
_C.TRAINER.DAEL.CONF_THRE = 0.95  # 置信度阈值
_C.TRAINER.DAEL.STRONG_TRANSFORMS = ()  # 强数据增强变换列表

######
# 域泛化（Domain Generalization, DG）
######
# CrossGrad
_C.TRAINER.CROSSGRAD = CN()
_C.TRAINER.CROSSGRAD.EPS_F = 1.0  # 域分类器 D 梯度的缩放参数
_C.TRAINER.CROSSGRAD.EPS_D = 1.0  # 标签分类器 F 梯度的缩放参数
_C.TRAINER.CROSSGRAD.ALPHA_F = 0.5  # 标签网络损失的平衡权重
_C.TRAINER.CROSSGRAD.ALPHA_D = 0.5  # 域网络损失的平衡权重
# DDAIG（Diversify Domain-Aware Image Generation）
_C.TRAINER.DDAIG = CN()
_C.TRAINER.DDAIG.G_ARCH = ""  # 生成器的架构
_C.TRAINER.DDAIG.LMDA = 0.3  # 扰动权重
_C.TRAINER.DDAIG.CLAMP = False  # 是否限制扰动值
_C.TRAINER.DDAIG.CLAMP_MIN = -1.0  # 扰动值的最小值
_C.TRAINER.DDAIG.CLAMP_MAX = 1.0  # 扰动值的最大值
_C.TRAINER.DDAIG.WARMUP = 0  # 预热迭代次数
_C.TRAINER.DDAIG.ALPHA = 0.5  # 损失的平衡权重
# DAELDG（DAEL 的域泛化版本）
_C.TRAINER.DAELDG = CN()
_C.TRAINER.DAELDG.WEIGHT_U = 0.5  # 未标记数据损失的权重
_C.TRAINER.DAELDG.CONF_THRE = 0.95  # 置信度阈值
_C.TRAINER.DAELDG.STRONG_TRANSFORMS = ()  # 强数据增强变换列表
# DOMAINMIX（域混合）
_C.TRAINER.DOMAINMIX = CN()
_C.TRAINER.DOMAINMIX.TYPE = "crossdomain"  # 混合类型
_C.TRAINER.DOMAINMIX.ALPHA = 1.0  # Beta 分布的 alpha 参数
_C.TRAINER.DOMAINMIX.BETA = 1.0  # Beta 分布的 beta 参数

######
# 半监督学习（Semi-Supervised Learning, SSL）
######
# EntMin（Entropy Minimization）
_C.TRAINER.ENTMIN = CN()
_C.TRAINER.ENTMIN.LMDA = 1e-3  # 熵损失的权重
# Mean Teacher（平均教师）
_C.TRAINER.MEANTEACHER = CN()
_C.TRAINER.MEANTEACHER.WEIGHT_U = 1.0  # 未标记数据损失的权重
_C.TRAINER.MEANTEACHER.EMA_ALPHA = 0.999  # 指数移动平均的平滑系数
_C.TRAINER.MEANTEACHER.RAMPUP = 5  # 用于增加损失权重的 epoch 数
# MixMatch
_C.TRAINER.MIXMATCH = CN()
_C.TRAINER.MIXMATCH.WEIGHT_U = 100.0  # 未标记数据损失的权重
_C.TRAINER.MIXMATCH.TEMP = 2.0  # 用于锐化概率分布的温度参数
_C.TRAINER.MIXMATCH.MIXUP_BETA = 0.75  # Mixup 的 Beta 分布参数
_C.TRAINER.MIXMATCH.RAMPUP = 20000  # 用于增加损失权重的步数
# FixMatch
_C.TRAINER.FIXMATCH = CN()
_C.TRAINER.FIXMATCH.WEIGHT_U = 1.0  # 未标记数据损失的权重
_C.TRAINER.FIXMATCH.CONF_THRE = 0.95  # 置信度阈值
_C.TRAINER.FIXMATCH.STRONG_TRANSFORMS = ()  # 强数据增强变换列表

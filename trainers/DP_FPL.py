import torch
import torch.nn as nn
from torch.nn import functional as F

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import math
import time
import logging
from utils.logger import require_global_logger

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    import os
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    
    # 优先检查是否指定了模型文件路径
    if hasattr(cfg.MODEL.BACKBONE, 'PATH') and cfg.MODEL.BACKBONE.PATH:
        model_path = cfg.MODEL.BACKBONE.PATH
        if os.path.isfile(model_path):
            logger = require_global_logger()
            logger.info(f"使用指定的模型路径: {model_path}")
        else:
            raise FileNotFoundError(f"指定的模型文件不存在: {model_path}")
    else:
        # 检查是否指定了下载缓存目录
        cache_dir = None
        if hasattr(cfg.MODEL.BACKBONE, 'CACHE_DIR') and cfg.MODEL.BACKBONE.CACHE_DIR:
            cache_dir = cfg.MODEL.BACKBONE.CACHE_DIR
        elif os.environ.get('CLIP_CACHE_DIR'):
            cache_dir = os.environ.get('CLIP_CACHE_DIR')
        
        # 如果指定了缓存目录，使用该目录下载
        if cache_dir:
            model_path = clip._download(url, root=cache_dir)
        else:
            model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'SEPFPL',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def orthogonalize(matrix):
    m = matrix.shape[1]
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col

def factorize_ctx(origin, rank):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        v = torch.normal(0, 1, size=(origin.shape[1], rank)).type(origin.dtype) # [ctx_dim, rank]
        u = torch.matmul(origin.to(device), v.to(device)) # [n_ctx, rank]
        orthogonalize(u)
        v = torch.matmul(origin.t().to(device), u.to(device)) # [ctx_dim, rank]
        orthogonalize(v)
        v = v.t() # [rank, ctx_dim]
        residual = origin.to(device) - torch.matmul(u.to(device), v.to(device)) # [n_ctx, ctx_dim]

    return (u, v, residual)

def compute_full_grad(left, right, dtype):
        left_w, left_g = left.data.type(dtype), left.grad.type(dtype) / 10.0
        right_w, right_g = right.data.type(dtype), right.grad.type(dtype) / 10.0

        left_g_right_w = torch.matmul(left_g, right_w)
        m1 = left_g_right_w + torch.matmul(left_w, right_g)
        m2 = torch.matmul(left_w, torch.matmul(left_w.T, left_g_right_w))

        return m1 + m2

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device) # [100,77,512] + [77, 512]

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg  # 保存cfg以便在forward方法中使用
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DP_FPL.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.factorization = cfg.FACTORIZATION
        self.rank = cfg.RANK
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # ========== 新增 cluster prompt 参数初始化（仅启用HCSE时使用） ==========
        use_hcse = (self.factorization in ['sepfpl', 'sepfpl_hcse'])            
        if use_hcse:
            cluster_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(cluster_ctx_vectors, std=0.02)
            self.cluster_ctx = nn.Parameter(cluster_ctx_vectors)
        
        # ========== 原有 global/local/uv prompt 初始化保持不变 ==========
        global_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # n_ctx = 16, ctx_dim = 512
        nn.init.normal_(global_ctx_vectors, std=0.02)
        self.global_ctx = nn.Parameter(global_ctx_vectors)

        # local u and v context vectors
        if self.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
            local_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # n_ctx = 16, ctx_dim = 512
            nn.init.normal_(local_ctx_vectors, std=0.02)
            self.local_ctx = nn.Parameter(local_ctx_vectors)
        if self.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
            local_u_ctx_vectors = torch.empty(n_ctx, self.rank, dtype=dtype)
            nn.init.normal_(local_u_ctx_vectors, std=0.02)
            self.local_u_ctx = nn.Parameter(local_u_ctx_vectors)
            local_v_ctx_vectors = torch.empty(self.rank, ctx_dim, dtype=dtype)
            nn.init.normal_(local_v_ctx_vectors, std=0.02)
            self.local_v_ctx = nn.Parameter(local_v_ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # prompts for each class, each of length 16

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("embedding", embedding)

        self.n_cls = n_cls # number of classes
        self.n_ctx = n_ctx # number of text encoder of text prompts = 16
        self.tokenized_prompts = tokenized_prompts # [n_cls, clip prompt token limit]
        self.name_lens = name_lens

    def forward(self):
        # 原有分支保持不变
        if self.factorization == 'promptfl':
            client_ctx = self.global_ctx
        elif self.factorization == 'fedotp':
            client_ctx = self.global_ctx + self.local_ctx
        elif self.factorization == 'fedpgp':
            client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx)
        else:
            local_u_ctx, local_v_ctx, residual = factorize_ctx(self.local_ctx.data, self.rank)
            self.local_u_ctx.data = local_u_ctx
            self.local_v_ctx.data = local_v_ctx
            if self.factorization == 'dplora':
                client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx)
            elif self.factorization == 'dpfpl':
                client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx) + residual
            elif self.factorization in ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
                client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx) + residual
            else:
                client_ctx = self.global_ctx
        # 仅在启用HCSE时叠加cluster_ctx（根据factorization自动判断）
        use_hcse = (self.factorization in ['sepfpl', 'sepfpl_hcse'])
        if getattr(self, 'cluster_ctx', None) is not None and use_hcse:
            client_ctx = client_ctx + self.cluster_ctx
        if client_ctx.dim() == 2:
            client_ctx = client_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        client_prompt = torch.cat(
            [
                self.token_prefix,
                client_ctx,
                self.token_suffix,
            ],
            dim=1,
        )
        return client_prompt


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)) # [batch, 3, 224, 224] -> [32, 512]

        client_prompt = self.prompt_learner() # [100,77,512] = [n_cls, clip prompt token limit, ctx_dim]
        tokenized_prompts = self.tokenized_prompts
        client_text_features = self.text_encoder(client_prompt, tokenized_prompts) # [100,512] = [n_cls, ctx_dim]

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        client_text_features = client_text_features / client_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity between local text features and image features
        sim = image_features @ client_text_features.t() # [batch, n_cls]
        local_image_logits = sim * self.logit_scale.exp()

        return local_image_logits


# @TRAINER_REGISTRY.register()
class SEPFPL(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DP_FPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """
        构建模型并初始化差分隐私相关参数
        
        主要步骤：
        1. 加载 CLIP 预训练模型
        2. 构建自定义 CLIP 模型（包含 PromptLearner）
        3. 冻结图像和文本编码器，仅训练 PromptLearner
        4. 初始化优化器和学习率调度器
        5. 计算差分隐私参数（RDP 噪声标准差）
        """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # ========== 初始化日志记录器 ==========
        # 注意：需确保外部已调用 init_logger_from_args 进行全局初始化
        self.logger = require_global_logger()
        
        # ========== 加载 CLIP 预训练模型 ==========
        self.logger.info(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        # CLIP 默认使用 fp16 精度，如果配置要求 fp32 或 amp，则转换为 fp32
        if cfg.TRAINER.DP_FPL.PREC == "fp32" or cfg.TRAINER.DP_FPL.PREC == "amp":
            clip_model.float()
        self.dtype = clip_model.dtype

        # ========== 构建自定义 CLIP 模型 ==========
        self.logger.info("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # ========== 冻结非 PromptLearner 的参数 ==========
        # 在联邦提示学习中，只训练 PromptLearner 的参数
        # 图像编码器和文本编码器保持冻结状态，不参与梯度更新
        self.logger.info("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # ========== 加载预训练权重（如果指定） ==========
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # ========== 将模型移动到指定设备 ==========
        self.model.to(self.device)

        # ========== 初始化优化器和学习率调度器 ==========
        # 注意：仅对 PromptLearner 的参数进行优化，其他模块参数已冻结
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        # ========== 计算差分隐私相关参数 ==========
        # 遍历所有客户端的数据加载器，获取最大 batch size 和最大 batch 数量
        # 这些值用于计算敏感度（sensitivity），进而确定噪声标准差
        max_batch_size = 0
        total_batches_per_round = 0
        for idx in range(0, cfg.DATASET.USERS):
            max_batch_size = max(max_batch_size, self.dm.fed_train_loader_x_dict[idx].batch_size)
            total_batches_per_round = max(total_batches_per_round, len(self.dm.fed_train_loader_x_dict[idx]))
        
        # ========== RDP (Rényi Differential Privacy) 噪声计算 ==========
        if cfg.NOISE > 0:
            # 从配置读取 RDP 参数
            # rdp_alpha: RDP 阶数 α，控制隐私保证的严格程度（默认 2.0）
            rdp_alpha = getattr(cfg, 'RDP_ALPHA', 2.0)
            
            # 计算每个 batch 的 RDP 隐私预算
            # 将总隐私预算平均分配到所有训练轮次
            rdp_eps_per_batch = cfg.NOISE / cfg.OPTIM.ROUND
            
            # 根据 RDP 理论计算噪声标准差
            # RDP 高斯机制公式: ε_α = α / (2 * σ^2)
            # 反推得到: σ = sqrt(α / (2 * ε_α))
            rdp_sigma = math.sqrt(rdp_alpha / (2.0 * rdp_eps_per_batch))
            
            # 计算敏感度（sensitivity）
            # 敏感度 = 梯度裁剪阈值 / batch_size
            # 表示单个样本对梯度的影响上限
            sensitivity = cfg.NORM_THRESH / max_batch_size
            
            # 最终的噪声标准差 = RDP 噪声系数 × 敏感度
            self.std = rdp_sigma * sensitivity
                        
            # ========== SepFPL 时间适应隐私预算分配 ==========
            # 根据 factorization 类型自动判断是否启用时间适应隐私分配
            # sepfpl 和 sepfpl_time_adaptive 默认启用此机制
            use_time_adaptive = (cfg.FACTORIZATION in ['sepfpl', 'sepfpl_time_adaptive'])
            
            if use_time_adaptive:
                # 从配置读取时间适应参数
                # rdp_p: 时间适应幂次参数，控制隐私预算的分配策略（默认 1.05）
                # p > 1 表示后期轮次分配更多隐私预算，有利于模型收敛
                rdp_p = getattr(cfg, 'RDP_P', 1.05)
                total_rounds = cfg.OPTIM.ROUND  # 总训练轮数
                rdp_eps_tot = cfg.NOISE  # 总隐私预算
                
                # 预计算所有轮次的隐私预算分配
                # 时间适应分配公式: ε_t = ε_tot * (t^p) / (sum_{j=1}^T j^p)
                # 其中 t 是当前轮次，T 是总轮数，p 是幂次参数
                # 计算归一化分母: sum_{j=1}^T j^p
                denominator = sum(j ** rdp_p for j in range(1, total_rounds + 1))
                
                # 预计算每轮的隐私预算和对应的噪声标准差
                # 这样可以在训练过程中直接查表，避免重复计算
                self.rdp_eps_per_batch_list = []
                self.std_per_batch_list = []
                
                for t in range(1, total_rounds + 1):
                    # 计算第 t 轮的隐私预算
                    eps_t = rdp_eps_tot * (t ** rdp_p) / denominator
                    self.rdp_eps_per_batch_list.append(eps_t)
                    
                    # 根据该轮的隐私预算计算对应的噪声标准差
                    # 使用相同的 RDP 公式: σ = sqrt(α / (2 * ε_α))
                    sigma_t = math.sqrt(rdp_alpha / (2.0 * eps_t))
                    std_t = sigma_t * sensitivity
                    self.std_per_batch_list.append(std_t)
                
                # 初始化第一轮的噪声标准差
                # 后续轮次通过 update_std_for_round() 方法更新
                self.std = self.std_per_batch_list[0]
                
        # ========== 初始化训练时间跟踪 ==========
        # 用于记录每个 epoch 的开始时间，便于性能分析和日志记录
        self.epoch_start_time = None
    
    def update_std_for_round(self, round_idx):
        """根据当前轮次更新噪声标准差
        Args:
            round_idx: 当前轮次（从0开始）
        """
        if hasattr(self, 'std_per_batch_list') and self.std_per_batch_list is not None:
            if 0 <= round_idx < len(self.std_per_batch_list):
                self.std = self.std_per_batch_list[round_idx]
                # 每一轮都打印信息
                if hasattr(self, 'rdp_eps_per_batch_list'):
                    logger = getattr(self, 'logger', None)
                    if logger is None:
                        logger = require_global_logger()
                    logger.info(f"[RDP-sepfpl] 轮次 {round_idx + 1}: ε={self.rdp_eps_per_batch_list[round_idx]:.6f}, std={self.std:.6f}")
            else:
                logger = getattr(self, 'logger', None)
                if logger is None:
                    logger = require_global_logger()
                logger.warning(f"[RDP-sepfpl] 警告: 轮次 {round_idx + 1} 超出范围，使用最后一轮的标准差")
                self.std = self.std_per_batch_list[-1]
    
    def reset_epoch_timer(self):
        """重置epoch开始时间，应该在每个epoch开始时调用"""
        self.epoch_start_time = time.time()

    def forward_pass(self, batch):
        """
        前向传播和梯度处理
        
        主要步骤：
        1. 前向传播计算损失
        2. 反向传播计算梯度
        3. 梯度裁剪（限制梯度范数，满足差分隐私敏感度要求）
        4. 梯度加噪（添加高斯噪声，实现差分隐私保护）
        5. 计算完整梯度（对于低秩分解方法，从 u 和 v 的梯度恢复完整梯度）
        
        注意：不同 factorization 方法需要处理不同的参数：
        - promptfl: 仅处理 global_ctx
        - fedotp: 处理 global_ctx 和 local_ctx
        - fedpgp/dplora/dpfpl/sepfpl: 处理 global_ctx, local_u_ctx, local_v_ctx
        - sepfpl (HCSE): 额外处理 cluster_ctx
        """
        # ========== 前向传播和损失计算 ==========
        image, label = self.parse_batch_train(batch)
        logits = self.model(image)
        loss = F.cross_entropy(logits.float(), label)

        # ========== 反向传播计算梯度 ==========
        self.model_zero_grad()
        self.model_backward(loss)

        # 获取所有参数的字典，便于后续访问和修改梯度
        param_dict = dict(self.model.named_parameters())

        # ========== 差分隐私保护：梯度裁剪与加噪 ==========
        if self.cfg.NOISE > 0:
            # 梯度裁剪：限制 global_ctx 的梯度范数
            # 这是差分隐私的基础要求，确保敏感度（sensitivity）有界
            grad = param_dict['prompt_learner.global_ctx'].grad.data
            norm = grad.norm(2)  # 计算 L2 范数
            if norm > self.cfg.NORM_THRESH:
                # 如果梯度范数超过阈值，按比例缩放
                scale = self.cfg.NORM_THRESH / norm
                scale[scale>1] = 1  # 确保不会放大梯度
                param_dict['prompt_learner.global_ctx'].grad *= scale
            
            # ========== 根据不同的 factorization 方法处理相应参数 ==========
            if self.cfg.FACTORIZATION == 'promptfl':
                # PromptFL: 仅对 global_ctx 添加高斯噪声
                # 这是最简单的差分隐私实现，所有客户端共享同一个 global prompt
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.global_ctx'].grad += noise
            
            elif self.cfg.FACTORIZATION == 'fedotp':
                # FedOTP: 对 local_ctx 进行裁剪和加噪
                # FedOTP 使用 global_ctx + local_ctx 的组合，只对 local 部分加噪
                grad = param_dict['prompt_learner.local_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_ctx'].grad += noise
            
            elif self.cfg.FACTORIZATION in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
                # FedPGP/DPLoRA/DPFPL/SepFPL 系列: 对低秩分解参数进行裁剪和加噪
                # 这些方法将 local_ctx 分解为 local_u_ctx @ local_v_ctx
                # 只对分解后的参数加噪，可以减少通信成本和噪声影响
                
                # 处理 local_u_ctx: 裁剪 + 加噪
                grad = param_dict['prompt_learner.local_u_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_u_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_u_ctx'].grad += noise
                
                # 处理 local_v_ctx: 裁剪 + 加噪
                grad = param_dict['prompt_learner.local_v_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_v_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_v_ctx'].grad += noise
            
            # ========== SepFPL (HCSE): 对 cluster_ctx 进行裁剪和加噪 ==========
            # 仅在启用层次化类别语义编码（HCSE）时处理
            # cluster_ctx 用于捕获类别间的层次关系
            use_hcse_for_noise = getattr(self.cfg, 'USE_HCSE', None)
            if use_hcse_for_noise is None:
                # 自动判断：sepfpl 和 sepfpl_hcse 默认启用 HCSE
                use_hcse_for_noise = (self.cfg.FACTORIZATION in ['sepfpl', 'sepfpl_hcse'])
            
            if use_hcse_for_noise and 'prompt_learner.cluster_ctx' in param_dict and param_dict['prompt_learner.cluster_ctx'].grad is not None:
                grad = param_dict['prompt_learner.cluster_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.cluster_ctx'].grad *= scale
                # noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                # param_dict['prompt_learner.cluster_ctx'].grad += noise

        # ========== 计算完整梯度（用于低秩分解方法） ==========
        # 对于使用低秩分解的方法，需要从 u 和 v 的梯度恢复完整的 local_ctx 梯度
        # 这是为了在客户端本地更新时能够正确更新 local_ctx 参数
        # 公式: ∇local_ctx = ∇(u @ v) = ∇u @ v + u @ ∇v + u @ u^T @ ∇u @ v
        if self.cfg.FACTORIZATION in ['dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
            full_grad = compute_full_grad(
                param_dict['prompt_learner.local_u_ctx'], 
                param_dict['prompt_learner.local_v_ctx'], 
                self.dtype
            )
            full_grad = full_grad.type(self.dtype)
            param_dict['prompt_learner.local_ctx'].grad = full_grad

        # ========== 返回损失和准确率摘要 ==========
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        return loss_summary

    def train_forward(self, idx=-1, train_iter=None, current_batch=0, total_batches=0):
        """重写train_forward方法，添加loss_summary输出"""
        self.set_model_mode("train")
        
        batch = next(train_iter)
        loss_summary = self.forward_pass(batch)
        
        client_info = f"client {idx}" if idx >= 0 else "client unknown"
        batch_info = f"batch {current_batch + 1}/{total_batches}" if total_batches > 0 else f"batch {current_batch + 1}"
        logger = getattr(self, 'logger', None)
        if logger is None:
            logger = require_global_logger()
        # logger.info(f'{client_info} | {batch_info} : {loss_summary}')
        return loss_summary

    def backward_pass(self, avg_global_gradient, aggregated_cluster_gradient=None):
        # update global gradient
        param_dict = dict(self.model.named_parameters())
        param_dict['prompt_learner.global_ctx'].grad = avg_global_gradient
        if aggregated_cluster_gradient is not None and 'prompt_learner.cluster_ctx' in param_dict:
            param_dict['prompt_learner.cluster_ctx'].grad = aggregated_cluster_gradient
        # update
        self.model_update()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


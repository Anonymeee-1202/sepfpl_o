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
from utils.logger import get_logger, get_global_logger

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    import os
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    
    # 优先检查是否指定了模型文件路径
    if hasattr(cfg.MODEL.BACKBONE, 'PATH') and cfg.MODEL.BACKBONE.PATH:
        model_path = cfg.MODEL.BACKBONE.PATH
        if os.path.isfile(model_path):
            logger = get_global_logger() or get_logger('dp-fpl', log_dir='logs', log_to_file=False, log_to_console=True)
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

    design_details = {"trainer": 'DP_FPL',
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
        if self.factorization in ['fedotp', 'dplora', 'dpfpl', 'sepfpl']:
            local_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # n_ctx = 16, ctx_dim = 512
            nn.init.normal_(local_ctx_vectors, std=0.02)
            self.local_ctx = nn.Parameter(local_ctx_vectors)
        if self.factorization in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl']:
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
            client_ctx = client_ctx + 0.5 * self.cluster_ctx
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
class DP_FPL(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DP_FPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # 初始化logger，供类中其他方法使用
        # 参照 logger.py 中的命名规则：{dataset_name}_{factorization}_{rank}_{noise}_{seed}
        # 从 cfg 中提取信息
        try:
            dataset_name = str(cfg.DATASET.NAME).lower() if cfg.DATASET.NAME else 'unknown'
        except (AttributeError, KeyError):
            dataset_name = 'unknown'
        
        factorization = getattr(cfg, 'FACTORIZATION', 'unknown')
        rank = getattr(cfg, 'RANK', 'unknown')
        noise = getattr(cfg, 'NOISE', 'unknown')
        seed = getattr(cfg, 'SEED', 'unknown')
        
        # 构建日志名称：{dataset_name}_{factorization}_{rank}_{noise}_{seed}
        logger_name = f'{dataset_name}_{factorization}_{rank}_{noise}_{seed}'
        # 优先使用全局logger，如未初始化则按配置创建并注册
        self.logger = get_global_logger()
        if self.logger is None:
            self.logger = get_logger(logger_name, log_dir='logs', log_to_file=False, log_to_console=True)
        
        self.logger.info(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DP_FPL.PREC == "fp32" or cfg.TRAINER.DP_FPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        self.dtype = clip_model.dtype

        self.logger.info("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        # differential privacy parameters
        max_batch_size = 0
        total_batches_per_round = 0
        for idx in range(0, cfg.DATASET.USERS):
            max_batch_size = max(max_batch_size, self.dm.fed_train_loader_x_dict[idx].batch_size)
            total_batches_per_round = max(total_batches_per_round, len(self.dm.fed_train_loader_x_dict[idx]))
        if cfg.NOISE > 0:
            # ========== RDP (Rényi Differential Privacy) 实现 ==========
            # 从配置读取RDP参数
            rdp_alpha = getattr(cfg, 'RDP_ALPHA', 2.0)  # RDP阶数，默认2.0
            # 使用cfg.NOISE作为每个batch的RDP预算
            rdp_eps_per_batch = cfg.NOISE / cfg.OPTIM.ROUND
            
            # 使用RDP计算噪声标准差
            # 对于高斯机制: ε_α = α / (2 * σ^2)
            # 反推: σ = sqrt(α / (2 * ε_α))
            rdp_sigma = math.sqrt(rdp_alpha / (2.0 * rdp_eps_per_batch))
            sensitivity = cfg.NORM_THRESH / max_batch_size  # 敏感度
            self.std = rdp_sigma * sensitivity
                        
            # ========== sepfpl隐私预算分配实现（时间适应隐私分配）==========
            # 根据factorization自动判断：sepfpl/sepfpl_time_adaptive/sepfpl_hcse默认启用
            use_time_adaptive = (cfg.FACTORIZATION in ['sepfpl', 'sepfpl_time_adaptive'])
            if use_time_adaptive:
                # 从配置读取参数
                rdp_p = getattr(cfg, 'RDP_P', 1.05)  
                total_rounds = cfg.OPTIM.ROUND  # 总轮数
                
                rdp_eps_tot = cfg.NOISE
                
                # 预计算所有轮次的隐私预算分配
                # ε_t = ε_tot * (t^p) / (sum_{j=1}^T j^p)
                # 计算分母: sum_{j=1}^T j^p
                denominator = sum(j ** rdp_p for j in range(1, total_rounds + 1))
                
                # 预计算每轮的隐私预算和噪声标准差
                self.rdp_eps_per_batch_list = []
                self.std_per_batch_list = []
                
                for t in range(1, total_rounds + 1):
                    # 计算第t轮的隐私预算
                    eps_t = rdp_eps_tot * (t ** rdp_p) / denominator
                    self.rdp_eps_per_batch_list.append(eps_t)
                    
                    # 计算对应的噪声标准差: σ = sqrt(α / (2 * ε_α))
                    sigma_t = math.sqrt(rdp_alpha / (2.0 * eps_t))
                    std_t = sigma_t * sensitivity
                    self.std_per_batch_list.append(std_t)
                
                # 初始化第一轮的噪声标准差
                self.std = self.std_per_batch_list[0]
                
        
        # 初始化epoch开始时间跟踪
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
                        logger = get_global_logger() or get_logger('dp-fpl', log_dir='logs', log_to_file=False, log_to_console=True)
                    logger.info(f"[RDP-sepfpl] 轮次 {round_idx + 1}: ε={self.rdp_eps_per_batch_list[round_idx]:.6f}, std={self.std:.6f}")
            else:
                logger = getattr(self, 'logger', None)
                if logger is None:
                    # 如果 logger 未初始化，使用默认名称
                    logger = get_global_logger() or get_logger('dp-fpl', log_dir='logs', log_to_file=False, log_to_console=True)
                logger.warning(f"[RDP-sepfpl] 警告: 轮次 {round_idx + 1} 超出范围，使用最后一轮的标准差")
                self.std = self.std_per_batch_list[-1]
    
    def reset_epoch_timer(self):
        """重置epoch开始时间，应该在每个epoch开始时调用"""
        self.epoch_start_time = time.time()

    def forward_pass(self, batch):
        image, label = self.parse_batch_train(batch)
        logits = self.model(image)
        loss = F.cross_entropy(logits.float(), label)

        self.model_zero_grad()
        self.model_backward(loss)

        param_dict = dict(self.model.named_parameters())

        # 梯度裁剪与加噪（差分隐私保护）
        if self.cfg.NOISE > 0:
            # 裁剪global_ctx梯度
            grad = param_dict['prompt_learner.global_ctx'].grad.data
            norm = grad.norm(2)
            if norm > self.cfg.NORM_THRESH:
                scale = self.cfg.NORM_THRESH / norm
                scale[scale>1] = 1
                param_dict['prompt_learner.global_ctx'].grad *= scale
            
            # 根据不同的factorization方法，对相应参数进行裁剪和加噪
            if self.cfg.FACTORIZATION == 'promptfl':
                # promptfl: 仅对global_ctx加噪
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.global_ctx'].grad += noise
            
            elif self.cfg.FACTORIZATION == 'fedotp':
                # fedotp: 对local_ctx进行裁剪和加噪
                grad = param_dict['prompt_learner.local_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_ctx'].grad += noise
            
            elif self.cfg.FACTORIZATION in ['fedpgp', 'dplora', 'dpfpl', 'sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']:
                # fedpgp/dplora/dpfpl/sepfpl系列: 对local_u_ctx和local_v_ctx进行裁剪和加噪
                grad = param_dict['prompt_learner.local_u_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_u_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_u_ctx'].grad += noise
                # 处理local_v_ctx
                grad = param_dict['prompt_learner.local_v_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_v_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_v_ctx'].grad += noise
            
            # sepfpl: 对cluster_ctx进行裁剪和加噪（仅在启用HCSE时）
            use_hcse_for_noise = getattr(self.cfg, 'USE_HCSE', None)
            if use_hcse_for_noise is None:
                use_hcse_for_noise = (self.cfg.FACTORIZATION in ['sepfpl', 'sepfpl_hcse'])
            if use_hcse_for_noise and 'prompt_learner.cluster_ctx' in param_dict and param_dict['prompt_learner.cluster_ctx'].grad is not None:
                grad = param_dict['prompt_learner.cluster_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.cluster_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.cluster_ctx'].grad += noise

        if self.cfg.FACTORIZATION in ['dplora', 'dpfpl', 'sepfpl']:
            full_grad = compute_full_grad(param_dict['prompt_learner.local_u_ctx'], param_dict['prompt_learner.local_v_ctx'], self.dtype)
            full_grad = full_grad.type(self.dtype)
            param_dict['prompt_learner.local_ctx'].grad = full_grad

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
            logger = get_global_logger() or get_logger('dp-fpl', log_dir='logs', log_to_file=False, log_to_console=True)
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


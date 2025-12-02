import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from loguru import logger

_LOG_TAG_FIELD = "_log_tag"

_GLOBAL_LOGGER = None

def init_logger_from_args(
    args: Optional[Any] = None,
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    force_reinit: bool = False,
    context_extra: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    初始化全局 Logger。支持自动从 args 提取上下文、目录自动创建、日志轮转与压缩。
    
    Args:
        dataset_name, method_name, exp_name_override: 可通过 kwargs 传入以覆盖默认值。
    """
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is not None and not force_reinit:
        return _GLOBAL_LOGGER

    # --- 1. 上下文构建 (Priority: extra > args) ---
    keys = ["task_id", "factorization", "rank", "noise", "round", "num_users", "seed", "sepfpl_topk", "rdp_p"]
    context = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    if context_extra:
        context.update(context_extra)

    # --- 2. 确定元数据 (Priority: kwargs > context > args > default) ---
    def get_val(key, arg_attr, default, fallback=None):
        if key in kwargs and kwargs.get(key) is not None:
            return kwargs.get(key)
        value = context.get(arg_attr)
        if value not in [None, ""]:
            return value
        if args and hasattr(args, arg_attr):
            return getattr(args, arg_attr)
        if fallback is not None:
            return fallback(args) if callable(fallback) else fallback
        return default

    def infer_dataset_from_args(a):
        if a is None:
            return "default_ds"
        if hasattr(a, "dataset_config_file") and a.dataset_config_file:
            base = os.path.basename(a.dataset_config_file)
            name, _ = os.path.splitext(base)
            return name
        if hasattr(a, "dataset"):
            return a.dataset
        return "default_ds"

    dataset = get_val("dataset_name", "dataset", "default_ds", fallback=infer_dataset_from_args)
    method = get_val("method_name", "factorization", "default_method")
    rank_val = get_val("rank_value", "rank", None)
    noise_val = get_val("noise_value", "noise", None)
    users_val = (
        kwargs.get("num_users")
        or context.get("num_users")
        or getattr(args, "num_users", None)
        or getattr(args, "users", None)
        or None
    )
    task_id = context.get("task_id") or (getattr(args, "task_id", None) if args else None) or "task"
    
    # 获取 seed 值
    seed_val = (
        kwargs.get("seed")
        or context.get("seed")
        or (getattr(args, "seed", None) if args else None)
        or None
    )
    
    # 获取 sepfpl_topk 和 rdp_p（仅对 sepfpl 相关方法）
    sepfpl_methods = ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']
    is_sepfpl = method in sepfpl_methods
    
    sepfpl_topk_val = None
    rdp_p_val = None
    if is_sepfpl:
        sepfpl_topk_val = (
            kwargs.get("sepfpl_topk")
            or context.get("sepfpl_topk")
            or (getattr(args, "sepfpl_topk", None) if args else None)
        )
        rdp_p_val = (
            kwargs.get("rdp_p")
            or context.get("rdp_p")
            or (getattr(args, "rdp_p", None) if args else None)
        )
    
    # 获取 wandb_group（用于日志目录组织）
    wandb_group = (
        kwargs.get("wandb_group")
        or (getattr(args, "wandb_group", None) if args else None)
        or "default"
    )
    
    # 实验名逻辑: 指定 > (rank_noise) > logger_name
    default_exp = f"{getattr(args, 'rank', 'r')}_{getattr(args, 'noise', 'n')}" if args else "experiment"
    exp_name = kwargs.get("exp_name_override") or context.get("logger_name") or default_exp

    # 构建 log_tag，使用单字母标识符
    # 格式: {dataset}_{method}_r{rank}_n{noise}_s{seed}_k{topk}_p{rdp_p}_u{users}_{task_id}
    log_tag_parts = [dataset, method]
    
    # 添加 rank (r)
    if rank_val is not None:
        log_tag_parts.append(f"r{rank_val}")
    
    # 添加 noise (n)
    if noise_val is not None:
        log_tag_parts.append(f"n{noise_val}")
    
    # 添加 seed (s)
    if seed_val is not None:
        log_tag_parts.append(f"s{seed_val}")
    
    # 如果是 sepfpl 相关方法，添加 topk (k) 和 rdp_p (p)
    if is_sepfpl:
        if sepfpl_topk_val is not None:
            log_tag_parts.append(f"k{sepfpl_topk_val}")
        if rdp_p_val is not None:
            # 保留 rdp_p 中的点号，不替换
            log_tag_parts.append(f"p{rdp_p_val}")
    
    # 添加 users (u) 和 task_id
    if users_val is not None:
        log_tag_parts.append(f"u{users_val}")
    log_tag_parts.append(task_id)
    
    log_tag = "_".join(log_tag_parts)
    context[_LOG_TAG_FIELD] = log_tag

    # --- 3. 配置 Loguru ---
    logger.remove()
    
    # Console Sink
    if log_to_console:
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{extra[_log_tag]}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(sys.stdout, format=fmt, level=level, colorize=True)

    # File Sink
    if log_to_file:
        # 日志目录结构: log_dir / wandb_group / dataset / method
        log_path = Path(log_dir) / str(wandb_group) / str(dataset) / str(method)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 检查是否是 sepfpl 相关方法，如果是则添加 topk 和 rdp_p 参数
        sepfpl_methods = ['sepfpl', 'sepfpl_time_adaptive', 'sepfpl_hcse']
        is_sepfpl = method in sepfpl_methods
        
        # 基础文件名部分
        filename_parts = [rank_val, noise_val]
        
        # 如果是 sepfpl 相关方法，添加 topk 和 rdp_p
        if is_sepfpl:
            sepfpl_topk = (
                kwargs.get("sepfpl_topk")
                or context.get("sepfpl_topk")
                or (getattr(args, "sepfpl_topk", None) if args else None)
            )
            rdp_p = (
                kwargs.get("rdp_p")
                or context.get("rdp_p")
                or (getattr(args, "rdp_p", None) if args else None)
            )
            
            if sepfpl_topk is not None:
                filename_parts.append(sepfpl_topk)  # 直接添加数字，不加前缀
            if rdp_p is not None:
                # 直接使用 rdp_p 的字符串形式，保留原始格式（包含点号）
                filename_parts.append(str(rdp_p))
        
        # 添加 users 和 timestamp
        filename_parts.extend([users_val, timestamp])
        
        # 文件名格式: {rank}_{noise}_{[topkX]}_{[rdpY]}_{users}_{timestamp}.log
        log_file = log_path / f"{'_'.join(map(str, filename_parts))}.log"
        
        fmt_file = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level} | "
            "{extra[_log_tag]} | "
            "{message}"
        )
        logger.add(str(log_file), format=fmt_file, level=level, rotation="500 MB", retention="30 days", compression="zip")
        print(f"📋 Log file: {log_file}")

    # --- 4. 绑定上下文并保存 ---
    _GLOBAL_LOGGER = logger.bind(**context)
    return _GLOBAL_LOGGER

# --- 辅助接口 ---

def get_experiment_logger(*args, **kwargs):
    """init_logger_from_args 的别名"""
    return init_logger_from_args(*args, **kwargs)

def get_global_logger():
    """获取全局实例，未初始化返回 None"""
    return _GLOBAL_LOGGER

def require_global_logger():
    """获取全局实例，未初始化抛出异常"""
    if _GLOBAL_LOGGER is None:
        raise RuntimeError("Global logger not initialized. Call init_logger_from_args() first.")
    return _GLOBAL_LOGGER
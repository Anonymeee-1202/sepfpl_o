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
    keys = ["task_id", "factorization", "rank", "noise", "round", "num_users"]
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
    rank_val = get_val("rank_value", "rank", "r")
    noise_val = get_val("noise_value", "noise", "n")
    users_val = (
        kwargs.get("num_users")
        or context.get("num_users")
        or getattr(args, "num_users", None)
        or getattr(args, "users", None)
        or "users"
    )
    task_id = context.get("task_id") or (getattr(args, "task_id", None) if args else None) or "task"
    
    # 获取 wandb_group（用于日志目录组织）
    wandb_group = (
        kwargs.get("wandb_group")
        or (getattr(args, "wandb_group", None) if args else None)
        or "default"
    )
    
    # 实验名逻辑: 指定 > (rank_noise) > logger_name
    default_exp = f"{getattr(args, 'rank', 'r')}_{getattr(args, 'noise', 'n')}" if args else "experiment"
    exp_name = kwargs.get("exp_name_override") or context.get("logger_name") or default_exp

    log_tag = f"{dataset}_{method}_{rank_val}_{noise_val}_{users_val}_{task_id}"
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
        # 文件名格式: {rank}_{noise}_{users}_{timestamp}.log
        log_file = log_path / f"{rank_val}_{noise_val}_{users_val}_{timestamp}.log"
        
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
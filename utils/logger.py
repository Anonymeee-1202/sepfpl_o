"""
日志配置模块
统一管理项目中的日志输出，将print替换为logging
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name='dp-fpl', log_dir='logs', log_level=logging.INFO, 
                 log_to_file=True, log_to_console=True):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        log_level: 日志级别
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
    
    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_to_file:
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 日志文件名包含时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件已创建: {log_file}")
    
    return logger


_global_logger = None


def set_global_logger(logger: logging.Logger):
    """
    注册全局日志记录器，供项目内统一使用。
    """
    global _global_logger
    _global_logger = logger
    return _global_logger


def get_global_logger():
    """
    获取已经注册的全局日志记录器，若不存在则返回None。
    """
    return _global_logger


def get_logger(name='dp-fpl', log_dir='logs', log_level=logging.INFO,
               log_to_file=True, log_to_console=True):
    """
    获取项目使用的日志记录器。
    若已注册全局logger，则直接返回全局实例；否则按指定配置创建/获取。
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        log_level: 日志级别
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
    
    Returns:
        logger: 配置好的日志记录器
    """
    global _global_logger
    if _global_logger is not None:
        return _global_logger
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger = setup_logger(name, log_dir, log_level, log_to_file, log_to_console)
    _global_logger = logger
    return logger


def init_logger_from_args(args=None, log_dir='logs', log_to_file=True, log_to_console=True):
    """
    根据命令行参数初始化日志记录器
    
    Args:
        args: argparse.Namespace对象，包含配置参数
        log_dir: 日志文件保存目录
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 如果提供了args，可以从args中提取信息用于日志文件名
    if args is not None:
        # 尝试从args中提取数据集名称等信息
        dataset_name = 'unknown'
        if hasattr(args, 'dataset_config_file') and args.dataset_config_file:
            dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
        elif hasattr(args, 'dataset') and args.dataset:
            dataset_name = args.dataset
        
        # 提取参数信息，参考 federated_main.py 中的命名规则：acc_{factorization}_{rank}_{noise}_{seed}
        factorization = getattr(args, 'factorization', 'unknown')
        rank = getattr(args, 'rank', 'unknown')
        noise = getattr(args, 'noise', 'unknown')
        seed = getattr(args, 'seed', 'unknown')
        
        # 构建日志名称：{dataset_name}_{factorization}_{rank}_{noise}_{seed}
        # 与 pickle 文件命名规则保持一致
        name = f'{dataset_name}_{factorization}_{rank}_{noise}_{seed}'
    else:
        name = 'unknown'
    
    # 使用setup_logger而不是get_logger，确保每次都能创建新的logger（支持不同的factorization）
    # 注意：由于setup_logger会在文件名中添加时间戳，所以即使name相同，每次运行也会创建新的日志文件
    logger = setup_logger(name, log_dir, logging.INFO, log_to_file, log_to_console)
    set_global_logger(logger)
    return logger


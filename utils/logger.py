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


# 全局日志记录器实例
_logger = None


def get_logger(name='dp-fpl', log_dir='logs', log_level=logging.INFO, 
               log_to_file=True, log_to_console=True):
    """
    获取全局日志记录器（单例模式）
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        log_level: 日志级别
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
    
    Returns:
        logger: 配置好的日志记录器
    """
    global _logger
    if _logger is None:
        _logger = setup_logger(name, log_dir, log_level, log_to_file, log_to_console)
    return _logger


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
        
        name = f'dp-fpl-{dataset_name}'
    else:
        name = 'dp-fpl'
    
    return get_logger(name, log_dir, logging.INFO, log_to_file, log_to_console)


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
                 log_to_file=True, log_to_console=True, 
                 context_info=None):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        log_level: 日志级别
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
        context_info: 上下文信息字典（包含task_id、dataset、factorization等），用于在日志中显示
    
    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 根据是否有上下文信息选择不同的日志格式
    if context_info and log_to_console:
        # 控制台格式：包含简短上下文信息（更易读）
        console_formatter = logging.Formatter(
            '%(asctime)s | [%(levelname)s] | %(message)s',
            datefmt='%H:%M:%S'
        )
        # 文件格式：完整信息
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # 标准格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = formatter
        file_formatter = formatter
    
    # 存储上下文信息到logger，供后续使用
    if context_info:
        logger.context_info = context_info
    
    # 控制台输出
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
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
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"================================================")
        logger.info(f"日志文件已创建: {log_file}")
        logger.info(f"================================================")
    
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
        
        # 提取参数信息，参考 federated_main.py 中的命名规则：acc_{factorization}_{rank}_{noise}_{seed}_{num_users}
        factorization = getattr(args, 'factorization', 'unknown')
        rank = getattr(args, 'rank', 'unknown')
        noise = getattr(args, 'noise', 'unknown')
        seed = getattr(args, 'seed', 'unknown')
        num_users = getattr(args, 'num_users', 'unknown')
        
        # 构建日志名称
        name = f'{rank}_{noise}_{seed}_{num_users}'
    else:
        dataset_name = 'unknown'
        factorization = 'unknown'
        rank = 'unknown'
        noise = 'unknown'
        seed = 'unknown'
        num_users = 'unknown'
        name = f'{rank}_{noise}_{seed}_{num_users}'

    dataset_log_dir = os.path.join(log_dir, dataset_name, str(factorization))
    
    # 准备上下文信息，用于日志格式和摘要显示
    context_info = None
    if args is not None:
        task_id = getattr(args, 'task_id', None)
        num_users = str(num_users)
        partition = getattr(args, 'partition', 'noniid-labeldir')
        round_num = getattr(args, 'round', 'unknown')
        
        context_info = {
            'task_id': task_id,
            'dataset': dataset_name,
            'factorization': factorization,
            'rank': rank,
            'noise': noise,
            'seed': seed,
            'num_users': num_users,
            'partition': partition,
            'round': round_num,
        }
    
    # 使用setup_logger而不是get_logger，确保每次都能创建新的logger（支持不同的factorization）
    # 注意：由于setup_logger会在文件名中添加时间戳，所以即使name相同，每次运行也会创建新的日志文件
    logger = setup_logger(name, dataset_log_dir, logging.INFO, log_to_file, log_to_console, context_info=context_info)
    
    # 打印清晰的实验配置摘要（在控制台和日志文件中都显示）
    if context_info:
        logger.info("")
        logger.info("=" * 70)
        logger.info("📋 实验配置摘要")
        logger.info("=" * 70)
        if context_info['task_id']:
            logger.info(f"  Task ID:      {context_info['task_id']}")
        logger.info(f"  数据集:       {context_info['dataset']}")
        logger.info(f"  模型方法:     {context_info['factorization']}")
        logger.info(f"  Rank:         {context_info['rank']}")
        logger.info(f"  噪声级别:     {context_info['noise']}")
        logger.info(f"  随机种子:     {context_info['seed']}")
        logger.info(f"  客户端数量:   {context_info['num_users']}")
        logger.info(f"  数据划分:     {context_info['partition']}")
        logger.info(f"  训练轮次:     {context_info['round']}")
        logger.info("=" * 70)
        logger.info("")
    
    set_global_logger(logger)
    return logger


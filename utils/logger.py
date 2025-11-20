"""
日志配置模块（基于 Loguru 改进版）
统一管理项目中的日志输出，提供更强大的功能和更好的用户体验
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from loguru import logger as _loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    # 如果 loguru 不可用，回退到标准 logging
    import logging
    _loguru_logger = None


class LoggerAdapter:
    """
    Loguru Logger 适配器，提供与标准 logging.Logger 兼容的接口
    同时支持 Loguru 的高级功能
    """
    def __init__(self, loguru_logger, context_info: Optional[Dict[str, Any]] = None):
        self._logger = loguru_logger
        self.context_info = context_info or {}
        self._console_sink_id = None
        self._file_sink_id = None
        self.log_file = None
    
    def _format_message(self, msg: str) -> str:
        """格式化消息，包含上下文信息"""
        if self.context_info:
            # 只显示关键上下文信息
            key_fields = ['task_id', 'dataset', 'factorization']
            context_items = []
            for k in key_fields:
                if k in self.context_info and self.context_info[k]:
                    context_items.append(f"{k}={self.context_info[k]}")
            if context_items:
                return f"[{' | '.join(context_items)}] {msg}"
        return msg
    
    def debug(self, msg: str, *args, **kwargs):
        """记录调试信息"""
        formatted_msg = self._format_message(msg)
        self._logger.debug(formatted_msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """记录信息"""
        formatted_msg = self._format_message(msg)
        self._logger.info(formatted_msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """记录警告"""
        formatted_msg = self._format_message(msg)
        self._logger.warning(formatted_msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """记录错误"""
        formatted_msg = self._format_message(msg)
        self._logger.error(formatted_msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """记录严重错误"""
        formatted_msg = self._format_message(msg)
        self._logger.critical(formatted_msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, exc_info=True, **kwargs):
        """记录异常信息"""
        formatted_msg = self._format_message(msg)
        self._logger.exception(formatted_msg, *args, **kwargs)
    
    def setLevel(self, level):
        """设置日志级别（兼容标准 logging 接口）"""
        # Loguru 的级别在 add() 时设置，这里只是兼容接口
        pass
    
    def addHandler(self, handler):
        """兼容接口，Loguru 不需要 handler"""
        pass
    
    @property
    def handlers(self):
        """兼容接口"""
        return []


def _convert_log_level(level):
    """将标准 logging 级别转换为 Loguru 级别"""
    if isinstance(level, str):
        return level.upper()
    level_map = {
        10: "DEBUG",
        20: "INFO",
        30: "WARNING",
        40: "ERROR",
        50: "CRITICAL"
    }
    return level_map.get(level, "INFO")


def setup_logger(name='dp-fpl', log_dir='logs', log_level=20, 
                 log_to_file=True, log_to_console=True, 
                 context_info=None):
    """
    设置日志记录器（基于 Loguru）
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        log_level: 日志级别（10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL）
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
        context_info: 上下文信息字典（包含task_id、dataset、factorization等），用于在日志中显示
    
    Returns:
        logger: 配置好的日志记录器（LoggerAdapter 实例）
    """
    if not LOGURU_AVAILABLE:
        # 回退到标准 logging
        import logging
        return _setup_standard_logger(name, log_dir, log_level, log_to_file, log_to_console, context_info)
    
    # Loguru 是单例，我们使用配置来区分不同的 logger
    # 移除默认 handler（只在第一次调用时）
    try:
        _loguru_logger.remove()  # 移除默认的 stderr handler
    except ValueError:
        pass  # 如果已经移除过，忽略错误
    
    log_level_str = _convert_log_level(log_level)
    
    # 控制台输出（带颜色和格式化）
    console_sink_id = None
    if log_to_console:
        # 简洁的控制台格式
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )
        
        console_sink_id = _loguru_logger.add(
            sys.stdout,
            format=console_format,
            level=log_level_str,
            colorize=True,
        )
    
    # 文件输出
    log_file = None
    file_sink_id = None
    if log_to_file:
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 日志文件名包含时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'{name}_{timestamp}.log'
        
        # 文件格式（更详细，包含完整信息）
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name} | "
            "{message}"
        )
        
        file_sink_id = _loguru_logger.add(
            str(log_file),
            format=file_format,
            level=log_level_str,
            encoding="utf-8",
            rotation="500 MB",  # 文件大小超过 500MB 时轮转
            retention="30 days",  # 保留 30 天的日志
            compression="zip",  # 压缩旧日志
            backtrace=True,  # 显示完整的异常堆栈
            diagnose=True,  # 显示变量值
        )
    
    # 绑定上下文信息到 logger
    bound_logger = _loguru_logger.bind(name=name)
    if context_info:
        bound_logger = bound_logger.bind(**context_info)
    
    # 创建适配器
    adapter = LoggerAdapter(bound_logger, context_info)
    adapter.log_file = log_file  # 保存日志文件路径
    adapter._console_sink_id = console_sink_id  # 保存 sink ID 以便后续清理
    adapter._file_sink_id = file_sink_id
    
    # 记录日志文件创建信息
    if log_file:
        adapter.info("=" * 70)
        adapter.info(f"日志文件已创建: {log_file}")
        adapter.info("=" * 70)
    
    return adapter


def _setup_standard_logger(name='dp-fpl', log_dir='logs', log_level=20, 
                           log_to_file=True, log_to_console=True, 
                           context_info=None):
    """
    回退方案：使用标准 logging（当 Loguru 不可用时）
    """
    import logging
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 根据是否有上下文信息选择不同的日志格式
    if context_info and log_to_console:
        console_formatter = logging.Formatter(
            '%(asctime)s | [%(levelname)s] | %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = formatter
        file_formatter = formatter
    
    # 存储上下文信息到logger
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
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
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


def set_global_logger(logger):
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


def get_logger(name='dp-fpl', log_dir='logs', log_level=20,
               log_to_file=True, log_to_console=True):
    """
    获取项目使用的日志记录器。
    若已注册全局logger，则直接返回全局实例；否则按指定配置创建/获取。
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        log_level: 日志级别（10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL）
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
    
    Returns:
        logger: 配置好的日志记录器
    """
    global _global_logger
    if _global_logger is not None:
        return _global_logger
    
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
        
        # 提取参数信息
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
    
    # 使用setup_logger创建新的logger
    logger = setup_logger(name, dataset_log_dir, 20, log_to_file, log_to_console, context_info=context_info)
    
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

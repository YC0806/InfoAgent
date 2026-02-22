# backend/utils/logger.py
"""
日志配置模块
提供统一的日志记录器配置
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# 日志格式配置
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "infogather",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    配置并返回 logger 实例
    
    Args:
        name: Logger 名称（通常使用模块名）
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 可选的日志文件路径
        console: 是否输出到控制台
        
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 创建 formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # 控制台 handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件 handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    获取指定模块的 logger
    
    Args:
        module_name: 模块名（通常传入 __name__）
        
    Returns:
        Logger 实例
        
    Example:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello World")
    """
    return logging.getLogger(module_name)


# 全局初始化（在应用启动时调用）
def init_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None
) -> None:
    """
    初始化全局日志配置
    
    Args:
        level: 全局日志级别
        log_dir: 日志文件目录
    """
    log_file = None
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_file = log_dir / f"infogather_{timestamp}.log"
    
    # 配置根 logger
    setup_logger(
        name="",
        level=level,
        log_file=log_file,
        console=True
    )
    
    # 禁用第三方库的冗余日志
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

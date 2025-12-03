import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(config_path='configs/logging.yaml', env_key='LOG_CFG'):
    """
    加载日志配置，支持环境变量覆盖
    
    Args:
        config_path: 默认配置文件路径
        env_key: 环境变量名，可指定其他配置文件
    """
    # 从环境变量获取配置路径
    path = os.getenv(env_key, config_path)

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 加载配置
    if os.path.exists(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 支持环境变量覆盖日志级别
        if log_level := os.getenv('LOG_LEVEL'):
            # 更新所有logger的level
            for logger_name in config.get('loggers', {}):
                config['loggers'][logger_name]['level'] = log_level
            if 'root' in config:
                config['root']['level'] = log_level
        
        logging.config.dictConfig(config)
    else:
        # 配置文件不存在时使用默认配置
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"日志配置文件 {path} 不存在，使用默认配置")
    
    return logging.getLogger(__name__)

def get_logger(name):
    """
    获取指定名称的logger（确保已加载配置）
    
    使用方法:
        from logger_loader import get_logger
        logger = get_logger('capture_service')
    """
    return logging.getLogger(name)

# 全局加载配置
setup_logging()
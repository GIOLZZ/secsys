import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import time
import yaml
import multiprocessing as mp

from capture_worker import CaptureWorker
from utils.logger import get_logger


class CaptureService:
    """视频采集服务管理器"""
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.redis_config = self.config['redis']
        self.streams = self.config['streams']
        self.processes = []
        
        self.logger = get_logger(__name__)
    
    def start(self):
        """启动所有视频流采集进程"""
        self.logger.info("启动视频采集服务...")
        
        for stream_cfg in self.streams:
            if not stream_cfg.get('enable', True):
                self.logger.info(f"跳过禁用流: {stream_cfg['stream_id']}")
                continue
            
            # 为每路流创建独立进程
            process = mp.Process(
                target=self._run_capture_worker,
                args=(stream_cfg, self.redis_config),
                name=f"Capture-{stream_cfg['stream_id']}"
            )
            process.daemon = True
            process.start()
            self.processes.append(process)
            
            self.logger.info(
                f"启动采集进程 [{process.pid}] - 流ID: {stream_cfg['stream_id']}"
            )
        
        # 主进程保持运行
        try:
            while True:
                self._monitor_processes()
                time.sleep(10)
        except KeyboardInterrupt:
            self.stop()
    
    def _run_capture_worker(self, stream_cfg, redis_cfg):
        """运行采集工作进程"""
        worker = CaptureWorker(stream_cfg, redis_cfg)
        worker.run()
    
    def _monitor_processes(self):
        """监控子进程状态"""
        for p in self.processes:
            if not p.is_alive():
                self.logger.error(f"进程 {p.name} 已退出，准备重启...")
                # 实际项目可添加重启逻辑
    
    def stop(self):
        """停止所有采集进程"""
        self.logger.info("正在停止采集服务...")
        for p in self.processes:
            p.terminate()
            p.join(timeout=5)
        self.logger.info("采集服务已停止")
    


if __name__ == '__main__':
    service = CaptureService('configs/push_str.yaml')
    service.start()
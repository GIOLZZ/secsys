import time
import cv2

from frame_publisher import FramePublisher
from utils.logger import get_logger


class CaptureWorker:
    """单路视频流采集工作进程"""
    def __init__(self, stream_cfg, redis_config):
        self.stream_id = stream_cfg['stream_id']
        self.protocol = stream_cfg['protocol']
        self.url = self._build_url(stream_cfg)
        self.fps = stream_cfg.get('fps', 25)
        self.frame_interval = 1.0 / self.fps
        
        # 初始化发布器
        self.publisher = FramePublisher(
            stream_id=self.stream_id,
            redis_config=redis_config,
            maxlen=stream_cfg.get('stream_maxlen', 100)
        )
        
        # 初始化日志
        self.logger = get_logger(f"Capture-{self.stream_id}")
        
        # 统计信息
        self.stats = {'frames_captured': 0, 'errors': 0}
    
    def _build_url(self, cfg):
        """构建视频源URL"""
        if cfg['protocol'] == 'rtsp':
            return cfg['url']
        elif cfg['protocol'] == 'usb':
            return int(cfg['device_id'])
        elif cfg['protocol'] == 'file':
            return cfg['file_path']
        elif cfg['protocol'] == 'gb28181':
            # 简化为RTSP模拟（实际需集成SIP协议栈）
            self.logger.warning("GB28181需集成SIP协议栈，此处使用模拟RTSP")
            return cfg.get('rtsp_url', 'rtsp://localhost:8554/test')
        else:
            raise ValueError(f"不支持的协议: {cfg['protocol']}")
    
    def run(self):
        """主循环：持续采集并推送帧"""
        self.logger.info(f"开始采集: {self.url}")
        
        while True:
            try:
                if self.protocol == 'file':
                    self._file_loop()
                else:
                    self._capture_loop()
            except Exception as e:
                self.logger.error(f"采集异常: {e}", exc_info=True)
                self.stats['errors'] += 1
                time.sleep(5)  # 异常后等待重试
    
    def _capture_loop(self):
        """视频采集循环"""
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频源: {self.url}")
            raise RuntimeError("视频源连接失败")
        
        # 设置缓冲区以降低延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        last_frame_time = time.time()
        s = 0
        while True:
            try:
                ret, frame = cap.read()
                s += 1
                if not ret:
                    self.logger.warning("视频流读取失败，尝试重连...")
                    break
                
                # FPS控制
                current_time = time.time()
                if current_time - last_frame_time < self.frame_interval:
                    continue
                last_frame_time = current_time
                
                # 推送帧到Redis
                self.publisher.publish_frame(frame)
                
                # 更新统计
                self.stats['frames_captured'] += 1
                if self.stats['frames_captured'] % 10 == 0:
                    self.logger.info(
                        f"已采集 {self.stats['frames_captured']} 帧"
                    )
                
            except Exception as e:
                self.logger.error(f"帧处理失败: {e}")
                break
        
        cap.release()
        self.logger.info("视频源已释放")
    
    def _file_loop(self):
        """视频文件采集循环"""
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频源: {self.url}")
            raise RuntimeError("视频源连接失败")
        
        # 设置缓冲区以降低延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("视频流读取失败，尝试重连...")
                    break
                
                # 推送帧到Redis
                self.publisher.publish_frame(frame)
                
                # 更新统计
                self.stats['frames_captured'] += 1
                if self.stats['frames_captured'] % 10 == 0:
                    self.logger.info(
                        f"已采集 {self.stats['frames_captured']} 帧"
                    )
                
                time.sleep(self.frame_interval)
                
            except Exception as e:
                self.logger.error(f"帧处理失败: {e}")
                break
        
        cap.release()
        self.logger.info("视频源已释放")
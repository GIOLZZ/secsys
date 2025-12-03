import pickle
import time
import redis
import cv2

from datetime import datetime
from utils.logger import get_logger


class FramePublisher:
    """帧数据发布到Redis"""
    def __init__(self, stream_id: str, redis_config: dict, maxlen: int=100):
        """
        Args:
            stream_id (str): 流ID
            redis_config (dict): Redis配置
            maxlen (int): 最大缓存帧数
        """
        self.stream_id = stream_id
        self.stream_key = f"stream:{stream_id}:frames"
        self.meta_key = f"stream:{stream_id}:meta"
        self.maxlen = maxlen
        
        # 连接Redis
        self.redis_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            password=redis_config.get('password'),
            db=redis_config.get('db', 0),
            decode_responses=False  # 二进制数据
        )
        
        self.logger = get_logger(f"Publisher-{stream_id}")
        self._init_stream()
    
    def _init_stream(self):
        """初始化流元数据"""
        meta = {
            'stream_id': self.stream_id,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        self.redis_client.hset(self.meta_key, mapping=meta)
        self.redis_client.expire(self.meta_key, 86400)  # 24小时过期
    
    def publish_frame(self, frame):
        """发布帧到Redis Stream"""
        # try:
        #     # 编码帧数据（JPEG压缩）
        #     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        #     frame_bytes = pickle.dumps({
        #         'data': buffer.tobytes(),
        #         'shape': frame.shape,
        #         'dtype': str(frame.dtype)
        #     })

        #     message = {
        #         'timestamp': int(time.time() * 1000),
        #         'frame_id': self._generate_frame_id(),
        #         'frame_data': frame_bytes
        #     }
            
        #     # 写入Redis Stream 自动限制长度
        #     self.redis_client.xadd(
        #         self.stream_key,
        #         message,
        #         maxlen=self.maxlen,
        #         approximate=True  # 性能优化
        #     )
            
        #     # 设置TTL 自动清理旧数据
        #     self.redis_client.expire(self.stream_key, 3600)

        # except Exception as e:
        #     self.logger.error(f"帧发布失败: {e}")

        try:
            # 直接序列化原始帧数据（零拷贝）
            frame_bytes = pickle.dumps({
                'data': frame.tobytes(),  # 原始字节
                'shape': frame.shape,
                'dtype': str(frame.dtype)
            })
            
            # 构建消息
            message = {
                'timestamp': int(time.time() * 1000),
                'frame_id': self._generate_frame_id(),
                'frame_data': frame_bytes,
                'encoded': b'false'  # 标记未编码
            }
            
            # 写入Redis Stream
            self.redis_client.xadd(
                self.stream_key,
                message,
                maxlen=self.maxlen,
                approximate=True
            )
            
            # 设置TTL
            self.redis_client.expire(self.stream_key, 3600)
            
        except Exception as ex:  # 修复：使用不同的变量名避免冲突
            self.logger.error(f"帧发布失败: {ex}", exc_info=True)

    def _generate_frame_id(self):
        """生成唯一帧ID"""
        return f"{self.stream_id}:{int(time.time() * 1000)}"
    
    def get_stream_info(self):
        """获取流信息"""
        info = self.redis_client.hgetall(self.meta_key)
        info['length'] = self.redis_client.xlen(self.stream_key)
        return info
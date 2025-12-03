import redis
import cv2
import time
import pickle
import numpy as np
from datetime import datetime


class FrameConsumer:
    """帧数据消费者（智能分析层）"""
    def __init__(self, stream_id, redis_config):
        self.stream_id = stream_id
        self.stream_key = f"stream:{stream_id}:frames"
        
        # 连接Redis
        self.redis_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            password=redis_config.get('password'),
            db=redis_config.get('db', 0),
            decode_responses=False
        )
        
        self.last_id = '0'  # 从开始消费
    
    def consume_latest_frame(self):
        """获取最新一帧"""
        # 读取最新消息
        messages = self.redis_client.xrevrange(
            self.stream_key,
            '+',  # 最新
            '-',  # 最旧
            count=1
        )
        
        if not messages:
            return None
        
        msg_id, msg_data = messages[0]
        return self._decode_frame(msg_data)
    
    def consume_next_frame(self, block_ms=1000):
        """阻塞读取下一帧"""
        # 使用XREAD阻塞读取
        streams = {self.stream_key: self.last_id}
        messages = self.redis_client.xread(
            streams,
            count=1,
            block=block_ms  # 阻塞等待
        )
        
        if not messages:
            return None
        
        # 解析消息
        stream_name, stream_messages = messages[0]
        msg_id, msg_data = stream_messages[0]
        self.last_id = msg_id
        
        return self._decode_frame(msg_data)
    
    def _decode_frame(self, msg_data):
        """解码帧数据"""
        # frame_bytes = msg_data[b'frame_data']
        # frame_info = pickle.loads(frame_bytes)
        
        # # 解码图像
        # nparr = np.frombuffer(frame_info['data'], np.uint8)
        # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # return {
        #     'frame': frame,
        #     'timestamp': int(msg_data[b'timestamp']),
        #     'frame_id': msg_data[b'frame_id'].decode(),
        #     'stream_id': self.stream_id
        # }

        try:
            frame_bytes = msg_data[b'frame_data']
            frame_info = pickle.loads(frame_bytes)
            
            # 直接从原始字节重建numpy数组
            frame = np.frombuffer(
                frame_info['data'],
                dtype=np.dtype(frame_info['dtype'])
            ).reshape(frame_info['shape'])
            
            return {
                'frame': frame,  # 可直接使用，无需解码
                'timestamp': int(msg_data[b'timestamp']),
                'frame_id': msg_data[b'frame_id'].decode(),
                'stream_id': self.stream_id
            }
        except Exception as ex:
            self.logger.error(f"帧解码失败: {ex}")
            return None

def simple_analyzer():
    """独立运行的分析程序"""
    redis_cfg = {'host': 'localhost', 'port': 6379, 'db': 0}
    
    # 可以同时消费多路流
    consumers = {
        # 'usb_cam': FrameConsumer('usb_cam', redis_cfg),
        # 'warehouse_cam': FrameConsumer('warehouse_cam', redis_cfg),
        'video_file': FrameConsumer('video_file', redis_cfg),
    }
    
    print("智能分析层已启动，等待帧数据...")
    
    while True:
        for stream_id, consumer in consumers.items():
            frame_data = consumer.consume_latest_frame()
            
            if frame_data:
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                # 这里插入你的AI推理代码
                # result = your_ai_model.infer(frame)
                
                print(f"[{datetime.now()}] {stream_id} - "
                      f"帧ID: {frame_data['frame_id']}, "
                      f"形状: {frame.shape}")
                
                # 显示画面（调试用）
                # cv2.imshow(stream_id, frame)
                cv2.imwrite('./test.png', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    simple_analyzer()
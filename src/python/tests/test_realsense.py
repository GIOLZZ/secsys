import cv2 


cap = cv2.VideoCapture('/home/cc/edge-agent/data/test_video/37b9517f5945f7792ea9d068fcbd4979.mp4')
if not cap.isOpened():
    print("视频源连接失败")
print("视频源连接")
while True:
    ret, frame = cap.read()
    print(frame)
    if not ret:
        print("视频流读取失败，尝试重连...")
        break

    cv2.imshow('Color', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # 创建流水线并配置
# pipeline = rs.pipeline()
# config = rs.config()

# # 配置要流式传输的格式和分辨率
# # 启用彩色流：640x480 分辨率，BGR格式
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# # 启用深度流：640x480 分辨率，Z16格式
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # 开始流式传输
# profile = pipeline.start(config)

# # 创建对齐对象（将深度帧与彩色帧对齐）
# align_to = rs.stream.color
# align = rs.align(align_to)

# try:
#     while True:
#         # 等待一组连贯的帧：深度帧和彩色帧
#         frames = pipeline.wait_for_frames()

#         # 将对齐深度帧到彩色帧
#         aligned_frames = align.process(frames)

#         # 获取对齐后的深度帧和彩色帧
#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         if not depth_frame or not color_frame:
#             continue

#         # 将图像转换为NumPy数组
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         # 将深度图像转换为彩色图以便可视化（可选）
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#         # 在同一窗口中水平堆叠显示彩色图像和深度图像
#         images = np.hstack((color_image, depth_colormap))
#         # cv2.namedWindow('RealSense', cv2.WINDOW_AORMAL)
#         cv2.imshow('RealSense', images)

#         # 按 'q' 或 ESC 键退出
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q') or key == 27:
#             break
# finally:
#     # 停止流式传输
#     pipeline.stop()
#     cv2.destroyAllWindows()
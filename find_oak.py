import depthai as dai
import cv2
import numpy as np

# 创建 pipeline
pipeline = dai.Pipeline()

# 创建彩色相机节点
cam_rgb = pipeline.create(dai.node.ColorCamera)
# 设置分辨率（可选：1080p, 4K等）
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# 设置预览尺寸
cam_rgb.setPreviewSize(640, 480)
# 设置帧率
cam_rgb.setFps(30)
# 设置相机ID（OAK-D-SR-POE有左右两个彩色相机，0是左边）
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# 创建输出节点，将图像发送到主机
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# 可选：添加深度相机（如果要用深度信息）
# mono_left = pipeline.create(dai.node.MonoCamera)
# mono_right = pipeline.create(dai.node.MonoCamera)
# stereo = pipeline.create(dai.node.StereoDepth)
# 
# mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
# mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
# 
# mono_left.out.link(stereo.left)
# mono_right.out.link(stereo.right)
# 
# xout_depth = pipeline.create(dai.node.XLinkOut)
# xout_depth.setStreamName("depth")
# stereo.depth.link(xout_depth.input)

# 连接到相机
device_info = dai.DeviceInfo("169.254.1.222")
with dai.Device(pipeline, device_info) as device:
    print("相机已启动，正在显示画面...")
    print("按 'q' 键退出")
    
    # 获取输出队列
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    # q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    while True:
        # 获取彩色图像
        in_rgb = q_rgb.tryGet()
        if in_rgb is not None:
            # 转换为OpenCV格式
            frame = in_rgb.getCvFrame()
            # 显示图像
            cv2.imshow("OAK-D-SR-POE Camera", frame)
        
        # 可选：显示深度图
        # in_depth = q_depth.tryGet()
        # if in_depth is not None:
        #     depth_frame = in_depth.getCvFrame()
        #     # 归一化深度图以便显示
        #     depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #     depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        #     cv2.imshow("Depth", depth_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) == ord('q'):
            break

# 清理
cv2.destroyAllWindows()
print("程序结束")
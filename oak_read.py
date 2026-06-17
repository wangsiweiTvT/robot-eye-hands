import cv2
import depthai as dai
#pip install depthai==2.31.1.0
import time
from datetime import datetime

# 创建管道
pipeline = dai.Pipeline()

# 配置参数
resolution = (1280, 800)  # THE_400_P分辨率
fps = 30  # 帧率
output_codec = 'MJPG'  # 更可靠的编码器
output_extension = '.avi'  # MJPG适合AVI容器

# 创建左右相机节点
left_camera = pipeline.create(dai.node.MonoCamera)
right_camera = pipeline.create(dai.node.MonoCamera)

# 创建 StereoDepth 节点
stereo = pipeline.create(dai.node.StereoDepth)

# 创建输出节点
xout_left = pipeline.create(dai.node.XLinkOut)
xout_right = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)

# 设置输出流名称
xout_left.setStreamName("left")
xout_right.setStreamName("right")
xout_depth.setStreamName("depth")

# 设置相机属性
left_camera.setBoardSocket(dai.CameraBoardSocket.LEFT)
left_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
left_camera.setFps(fps)

right_camera.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
right_camera.setFps(fps)

# 设置 StereoDepth 属性
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

# 连接节点
left_camera.out.link(stereo.left)
right_camera.out.link(stereo.right)
stereo.rectifiedLeft.link(xout_left.input)
stereo.rectifiedRight.link(xout_right.input)
stereo.depth.link(xout_depth.input)

time_now = datetime.now()
time_now = time_now.strftime('%Y-%m-%d_%H-%M-%S')
left_name = 'left_'+str(time_now)+output_extension
right_name = 'right_'+str(time_now)+output_extension
# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*output_codec)
out_left = cv2.VideoWriter(left_name, fourcc, fps, resolution, isColor=False)
out_right = cv2.VideoWriter(right_name, fourcc, fps, resolution, isColor=False)
device_info = dai.DeviceInfo("169.254.1.222")

if not out_left.isOpened() or not out_right.isOpened():
    raise Exception("无法创建视频文件，请检查编码器和路径")

try:
    with dai.Device(pipeline,device_info) as device:
        print(device.getConnectedCameraFeatures())
        print("设备启动，开始录制...")
        q_left = device.getOutputQueue(name="left", maxSize=4, blocking=True)
        q_right = device.getOutputQueue(name="right", maxSize=4, blocking=True)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=True)

        start_time = time.time()
        frame_count = 0

        while True:
            # 获取帧数据
            in_left = q_left.get()
            in_right = q_right.get()
            in_depth = q_depth.get()

            frame_left = in_left.getCvFrame()
            frame_right = in_right.getCvFrame()
            frame_depth = in_depth.getFrame()

            # 写入视频文件
            out_left.write(frame_left)
            out_right.write(frame_right)

            # 处理深度图显示
            frame_depth_normalized = cv2.normalize(frame_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_depth_colormap = cv2.applyColorMap(frame_depth_normalized, cv2.COLORMAP_JET)

            # 显示画面
            cv2.imshow("Left Camera", frame_left)
            cv2.imshow("Right Camera", frame_right)
            cv2.imshow("Depth Map", frame_depth_colormap)

            frame_count += 1
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed
            print(f"已录制: {frame_count}帧 | 实际FPS: {actual_fps:.1f}", end='\r')

            if cv2.waitKey(1) == ord('q'):
                break

finally:
    # 确保资源释放
    out_left.release()
    out_right.release()
    cv2.destroyAllWindows()
    print("\n录制完成，视频文件已保存")

    # 验证视频文件
    def verify_video(path):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return f"有效视频: {length}帧, {width}x{height}"
        return "视频文件损坏"

    print("左视频:", verify_video(f'left{output_extension}'))
    print("右视频:", verify_video(f'right{output_extension}'))
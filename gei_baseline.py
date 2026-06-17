import depthai as dai

# 创建管道
pipeline = dai.Pipeline()

# 创建相机节点
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_left = pipeline.create(dai.node.MonoCamera)
cam_right = pipeline.create(dai.node.MonoCamera)

# 设置相机分辨率
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# 创建输出流
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

xout_left = pipeline.create(dai.node.XLinkOut)
xout_left.setStreamName("left")
cam_left.out.link(xout_left.input)

xout_right = pipeline.create(dai.node.XLinkOut)
xout_right.setStreamName("right")
cam_right.out.link(xout_right.input)

# 连接设备并启动管道
with dai.Device(pipeline) as device:
    # 获取相机内参
    calib_data = device.readCalibration()
    intrinsics_rgb = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB)
    intrinsics_left = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT)
    intrinsics_right = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT)

    # 获取基线（左右相机之间的距离）
    baseline = calib_data.getBaselineDistance(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT)
    
    print("RGB相机内参:", intrinsics_rgb)
    print("左相机内参:", intrinsics_left)
    print("右相机内参:", intrinsics_right)
    print("左右相机基线:", baseline, "豪米")
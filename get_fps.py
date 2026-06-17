import depthai as dai

# 创建管道
pipeline = dai.Pipeline()

# 配置相机节点
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# 打印默认分辨率
print(f"默认分辨率: {cam_rgb.getResolution()}")

# 打印可用分辨率
print("可用分辨率:")
for res in cam_rgb.getResolutionSize():
    print(f"- {res}")

# 打印帧率范围
print(f"帧率范围: {cam_rgb.getFps()} fps (可设置)")
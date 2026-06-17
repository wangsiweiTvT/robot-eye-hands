import cv2
import numpy as np
import depthai as dai

# 初始化OAK相机
def initialize_oak_camera():
    pipeline = dai.Pipeline()

    # 创建左右相机节点
    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_right = pipeline.create(dai.node.MonoCamera)
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_right = pipeline.create(dai.node.XLinkOut)

    # 设置相机参数
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # 设置输出流
    xout_left.setStreamName("left")
    xout_right.setStreamName("right")

    # 连接相机和输出流
    cam_left.out.link(xout_left.input)
    cam_right.out.link(xout_right.input)

    return pipeline

# 计算视差图
def compute_disparity_map(left_image, right_image, ndisp=32, block_size=11):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=ndisp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=0,
        speckleRange=1
    )
    '''minDisparity:最小可能的差异值。正常情况下,它为零,但有时校正算法会移动图像,因此需要相应地调整此参数。
        numDisparities:disparity的搜索范围,即最大差异减去最小差异。该值始终大于零。在当前的实现中,这个参数必须可以被16整除。
        blockSize:块的线性大小。大小应该是奇数（因为块位于当前像素的中心）。更大的块大小意味着更平滑,但不太准确的视差图。较小的块大小会给出更详细的视差图,但算法找到错误对应的几率更高。一般在3到11之间。
        P1:控制视差平滑度的第一个参数,是相邻像素之间视差变化为1的惩罚。值越大,视差越平滑。
        P2:控制视差平滑度的第二个参数,是相邻像素之间视差变化超过1的惩罚。值越大,视差越平滑。该算法要求P2>P1。
        disp12MaxDiff:左右视差检查中允许的最大差异（以整数像素为单位）。将其设置为非正值以禁用检查。
        preFilterCap:预滤波图像像素的截断值。该算法首先计算每个像素的x方向的导数,并按[-preFilterCap,preFilterCap]间隔剪裁其值。结果值被传递到Birchfield-Tomasi像素代价函数。
        uniquenessRatio:最佳（最小）计算成本函数值应超过第二最佳值的百分比,满足此百分比的条件下才认为找到的匹配是正确的。通常,5-15范围内的值就足够好了。
        speckleWindowSize:考虑其噪声斑点的平滑差距区域的最大尺寸,并使之无效。把它设置为0以禁用斑点过滤。否则,将它设置在50-200范围内的某个地方。
        speckleRange:每个连接组件内的最大视差变化。如果进行斑点过滤,将参数设置为正值,它将被隐式地乘以16。通常,1或2就足够了。
        mode:默认情况下,它设置为false。若将其设置为MODE_HH,将运行完整的双过程动态规划算法。
    '''
    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    return disparity_map

# 转成深度图
def convert_disparity_to_depth(disparity_map, focal_length_px, baseline_mm):
    depth_map = (focal_length_px * baseline_mm) / (disparity_map + 1e-6)  # 避免除以零
    depth_map[disparity_map == 0] = 0  # 设置无效区域的深度为0
    return depth_map

# 主函数
def main():
    # 初始化OAK相机
    pipeline = initialize_oak_camera()

    # 启动相机
    with dai.Device(pipeline) as device:
        # 获取输出队列
        q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)

        # 获取相机内参和基线长度
        calib_data = device.readCalibration()
        intrinsics_left = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT)
        intrinsics_right = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT)
        baseline_mm = calib_data.getBaselineDistance(useSpecTranslation=True) * 10  # 转换为毫米

        # 提取焦距和主点
        focal_length_px = intrinsics_left[0][0]  # 焦距（fx）
        cx = intrinsics_left[0][2]  # 主点（cx）
        cy = intrinsics_left[1][2]  # 主点（cy）

        print(f"Focal Length (fx): {focal_length_px} px")
        print(f"Principal Point (cx, cy): ({cx}, {cy})")
        print(f"Baseline: {baseline_mm} mm")

        while True:
            # 获取左右图像
            left_frame = q_left.get().getCvFrame()
            right_frame = q_right.get().getCvFrame()

            # 计算视差图和深度图
            disparity_map = compute_disparity_map(left_frame, right_frame)
            depth_map = convert_disparity_to_depth(disparity_map, focal_length_px, baseline_mm)

            # 归一化深度图以便显示
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # 应用颜色映射
            depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

            # 显示左图像和深度图
            cv2.imshow("Left Camera", left_frame)
            cv2.imshow("Depth Map", depth_map_colored)

            # 按下 'q' 键退出
            if cv2.waitKey(1) == ord('q'):
                break

    # 释放资源
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
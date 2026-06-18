#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OAK相机标定板检测模块
用于手眼标定中的标定板位姿识别
支持: 棋盘格 (Chessboard) 和 Charuco 板
"""

import cv2
import numpy as np
import depthai as dai
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

device_info = dai.DeviceInfo("169.254.1.222")

# ==================== 数据结构 ====================
@dataclass
class BoardPose:
    """标定板位姿数据"""
    success: bool  # 是否成功检测
    x: float = 0.0  # X坐标 [mm]
    y: float = 0.0  # Y坐标 [mm]
    z: float = 0.0  # Z坐标 [mm]
    rx: float = 0.0  # 绕X轴旋转 [deg]
    ry: float = 0.0  # 绕Y轴旋转 [deg]
    rz: float = 0.0  # 绕Z轴旋转 [deg]
    rvec: np.ndarray = None  # 旋转向量 (原始数据)
    tvec: np.ndarray = None  # 平移向量 (原始数据)
    corners: np.ndarray = None  # 检测到的角点
    error: float = 0.0  # 重投影误差

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "x": self.x, "y": self.y, "z": self.z,
            "rx": self.rx, "ry": self.ry, "rz": self.rz,
            "error": self.error
        }


# ==================== OAK相机标定板检测器 ====================
class OAKBoardDetector:
    """
    OAK相机标定板检测器
    支持: 棋盘格 (Chessboard) 和 Charuco 板
    """

    def __init__(self,
                 board_type: str = "chessboard",
                 board_size: Tuple[int, int] = (7, 10),
                 square_size: float = 30.0,
                 camera_resolution: str = "1080p",
                 preview_size: Tuple[int, int] = (640, 480)):
        """
        初始化标定板检测器

        Args:
            board_type: 标定板类型 "chessboard" 或 "charuco"
            board_size: 棋盘格内角点数 (宽, 高)，如 (7, 10)
            square_size: 方格大小 [mm]
            camera_resolution: 相机分辨率 "1080p" 或 "4k"
            preview_size: 预览窗口大小
        """
        self.board_type = board_type
        self.board_size = board_size
        self.square_size = square_size
        self.preview_size = preview_size

        # 相机内参 (需要先标定或使用默认值)
        # 注意: 为了获得高精度，建议先用OAK标定程序标定相机内参
        self.camera_matrix = None
        self.dist_coeffs = None

        # OAK相机相关
        self.pipeline = None
        self.device = None
        self.queue = None

        # 标定板3D坐标 (以标定板中心为原点)
        self.object_points = self._create_board_points()

        # Charuco相关
        self.charuco_dict = None
        self.charuco_board = None
        self.charuco_params = None

        if board_type == "charuco":
            self._init_charuco()

        # 统计信息
        self.detection_count = 0
        self.success_count = 0

    def _create_board_points(self) -> np.ndarray:
        """
        创建标定板3D坐标 (以标定板中心为原点)

        Returns:
            (N, 3) 的3D点坐标数组
        """
        w, h = self.board_size
        objp = np.zeros((w * h, 3), dtype=np.float32)

        # 以标定板中心为原点
        offset_x = (w - 1) * self.square_size / 2.0
        offset_y = (h - 1) * self.square_size / 2.0

        for i in range(h):
            for j in range(w):
                idx = i * w + j
                objp[idx, 0] = j * self.square_size - offset_x
                objp[idx, 1] = i * self.square_size - offset_y
                objp[idx, 2] = 0.0

        return objp

    def _init_charuco(self):
        """初始化Charuco板"""
        # 使用5x5的ArUco字典
        self.charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

        # 创建Charuco板
        w, h = self.board_size
        self.charuco_board = cv2.aruco.CharucoBoard(
            (w, h),
            self.square_size,
            self.square_size * 0.6,  # 标记大小
            self.charuco_dict
        )

        # Charuco检测参数
        self.charuco_params = cv2.aruco.DetectorParameters()

    def setup_camera(self, camera_matrix: np.ndarray = None,
                     dist_coeffs: np.ndarray = None) -> bool:
        """
        设置相机内参 (如果不提供，会使用默认值)

        Args:
            camera_matrix: 3x3 相机内参矩阵
            dist_coeffs: 畸变系数
        """
        if camera_matrix is not None and dist_coeffs is not None:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            print("✓ 已加载相机内参")
        else:
            # 默认内参 (适用于OAK-D 1080p)
            # 注意: 这只是近似值，建议实际标定
            self.camera_matrix = np.array([
                [600.0, 0.0, 640.0],
                [0.0, 600.0, 360.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            print("⚠ 使用默认相机内参，建议进行相机标定以获得更高精度")

        return True

    def start_camera(self, camera_id: str = "auto") -> bool:
        """
        启动OAK相机

        Args:
            camera_id: 相机ID，通常使用 "auto" 或 "192.168.1.10" (网络连接)
        """
        print("正在启动OAK相机...")

        try:
            # 创建pipeline
            self.pipeline = dai.Pipeline()

            # 设置彩色相机节点
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(self.preview_size[0], self.preview_size[1])
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

            # 创建输出
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)

            # 连接设备
            self.device = dai.Device(self.pipeline,device_info)
            self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            print(f"✓ OAK相机启动成功")
            print(f"  预览分辨率: {self.preview_size}")

            return True

        except Exception as e:
            print(f"✗ OAK相机启动失败: {e}")
            return False

    def stop_camera(self):
        """停止OAK相机"""
        if self.device is not None:
            self.device.close()
            self.device = None
        self.queue = None
        self.pipeline = None
        print("OAK相机已关闭")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取一帧图像

        Returns:
            BGR图像 或 None
        """
        if self.queue is None:
            print("相机未启动")
            return None

        try:
            in_frame = self.queue.tryGet()
            if in_frame is not None:
                return in_frame.getCvFrame()
            return None
        except Exception as e:
            print(f"获取图像失败: {e}")
            return None

    def detect_chessboard(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        检测棋盘格

        Args:
            frame: BGR图像

        Returns:
            (是否检测成功, 角点坐标)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        w, h = self.board_size
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        if ret:
            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return ret, corners

    def detect_charuco(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        检测Charuco板

        Args:
            frame: BGR图像

        Returns:
            (是否检测成功, 角点坐标)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测ArUco标记
        detector = cv2.aruco.ArucoDetector(self.charuco_dict, self.charuco_params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) < 4:
            return False, None

        # 检测Charuco角点
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.charuco_board
        )

        if ret > 3:  # 至少需要4个角点
            return True, charuco_corners
        else:
            return False, None

    def compute_pose(self, corners: np.ndarray) -> BoardPose:
        """
        计算标定板位姿

        Args:
            corners: 检测到的角点坐标

        Returns:
            BoardPose 对象
        """
        if corners is None or len(corners) < 4:
            return BoardPose(success=False)

        # 使用solvePnP计算位姿
        # 注意: 需要根据实际检测到的角点数量调整object_points
        n_points = len(corners)
        if n_points != len(self.object_points):
            # 如果角点数量不匹配，取前n_points个
            obj_points = self.object_points[:n_points]
        else:
            obj_points = self.object_points

        # 转换为合适的格式
        if self.board_type == "charuco":
            # Charuco返回的是 (N, 1, 2) 格式
            corners_2d = corners.reshape(-1, 2)
        else:
            # 棋盘格返回的是 (N, 1, 2) 格式
            corners_2d = corners.reshape(-1, 2)

        # 求解PnP
        ret, rvec, tvec = cv2.solvePnP(
            obj_points.astype(np.float32),
            corners_2d.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ret:
            return BoardPose(success=False)

        # 计算重投影误差
        projected_points, _ = cv2.projectPoints(
            obj_points.astype(np.float32),
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        error = np.mean(np.linalg.norm(
            corners_2d.astype(np.float32) - projected_points.reshape(-1, 2),
            axis=1
        ))

        # 提取平移 (位置)
        x, y, z = tvec.flatten() * 1.0  # OAK-D的单位是mm

        # 旋转向量转欧拉角
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rx, ry, rz = self._rotation_matrix_to_euler(rotation_matrix)

        return BoardPose(
            success=True,
            x=x, y=y, z=z,
            rx=rx, ry=ry, rz=rz,
            rvec=rvec,
            tvec=tvec,
            corners=corners,
            error=error
        )

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        旋转矩阵转欧拉角 (XYZ顺序)

        Args:
            R: 3x3 旋转矩阵

        Returns:
            (rx, ry, rz) 角度 [度]
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        return np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)

    def detect_and_pose(self, frame: np.ndarray) -> BoardPose:
        """
        检测标定板并计算位姿 (主接口)

        Args:
            frame: BGR图像

        Returns:
            BoardPose 对象
        """
        self.detection_count += 1

        # 根据类型选择检测方法
        if self.board_type == "charuco":
            ret, corners = self.detect_charuco(frame)
        else:
            ret, corners = self.detect_chessboard(frame)

        if not ret or corners is None:
            return BoardPose(success=False)

        # 计算位姿
        pose = self.compute_pose(corners)
        if pose.success:
            self.success_count += 1

        # 可视化
        self._draw_board(frame, corners, pose)

        return pose

    def _draw_board(self, frame: np.ndarray, corners: np.ndarray, pose: BoardPose):
        """在图像上绘制检测结果"""
        if corners is not None:
            # 绘制角点
            if self.board_type == "charuco":
                cv2.aruco.drawDetectedCornersCharuco(frame, corners)
            else:
                cv2.drawChessboardCorners(frame, self.board_size, corners, True)

        if pose.success:
            # 绘制坐标系轴 (在标定板中心)
            if pose.rvec is not None and pose.tvec is not None:
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                  pose.rvec, pose.tvec, self.square_size * 2)

            # 显示位姿信息
            info = f"X={pose.x:.1f} Y={pose.y:.1f} Z={pose.z:.1f}mm"
            info += f" | Rx={pose.rx:.1f} Ry={pose.ry:.1f} Rz={pose.rz:.1f}°"
            info += f" | err={pose.error:.2f}px"

            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # 显示检测统计
            rate = self.success_count / max(self.detection_count, 1) * 100
            status = f"Detect: {self.detection_count} | Success: {self.success_count} | Rate: {rate:.1f}%"
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

    def get_pose_for_calibration(self, timeout: float = 30.0) -> Optional[BoardPose]:
        """
        获取用于手眼标定的位姿数据 (持续检测直到成功)

        Args:
            timeout: 超时时间 [秒]

        Returns:
            BoardPose 对象 或 None
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            pose = self.detect_and_pose(frame)

            # 显示实时画面
            cv2.imshow("Board Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break

            if pose.success:
                print(f"\r✓ 检测成功: {pose.to_dict()}", end="")
                return pose
            else:
                print(f"\r⏳ 检测中... (尝试 {self.detection_count})", end="")

        print("\n✗ 检测超时")
        return None


# ==================== 独立测试函数 ====================
def test_board_detection():
    """独立测试标定板检测"""
    print("=" * 60)
    print("OAK相机标定板检测测试")
    print("=" * 60)

    # 1. 创建检测器
    detector = OAKBoardDetector(
        board_type="chessboard",  # 或 "charuco"
        board_size=(8, 11),  # 棋盘格内角点数
        square_size=30.0,  # 方格大小 [mm]
        preview_size=(640, 480)
    )

    # 2. 设置相机内参 (如果有标定好的参数，请替换)
    # 注意: 这里的默认值精度有限，建议先用OAK标定程序标定
    detector.setup_camera()

    # 3. 启动相机
    if not detector.start_camera():
        print("相机启动失败，请检查连接")
        return

    print("\n" + "-" * 60)
    print("操作提示:")
    print("  - 将标定板放在相机视野内")
    print("  - 按 's' 键保存当前位姿数据")
    print("  - 按 'r' 键重置统计")
    print("  - 按 'q' 或 ESC 键退出")
    print("-" * 60)

    # 4. 测试循环
    saved_poses = []
    frame_count = 0

    try:
        while True:
            # 获取帧
            frame = detector.get_frame()
            if frame is None:
                continue

            frame_count += 1

            # 检测并计算位姿
            pose = detector.detect_and_pose(frame)

            # 显示
            cv2.imshow("Board Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and pose.success:
                # 保存位姿
                saved_poses.append(pose.to_dict())
                print(f"\n✓ 已保存位姿 #{len(saved_poses)}: {pose.to_dict()}")

            elif key == ord('r'):
                # 重置统计
                detector.detection_count = 0
                detector.success_count = 0
                saved_poses = []
                print("\n已重置统计和保存的数据")

            elif key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        # 5. 关闭相机
        detector.stop_camera()
        cv2.destroyAllWindows()

    # 6. 显示统计结果
    print("\n" + "=" * 60)
    print("测试结果统计:")
    print(f"  总帧数: {frame_count}")
    print(f"  检测尝试: {detector.detection_count}")
    print(f"  检测成功: {detector.success_count}")
    print(f"  成功率: {detector.success_count / max(detector.detection_count, 1) * 100:.1f}%")
    print(f"  保存位姿数: {len(saved_poses)}")

    if saved_poses:
        print("\n保存的位姿数据:")
        for i, p in enumerate(saved_poses):
            print(f"  [{i + 1}] X={p['x']:.2f} Y={p['y']:.2f} Z={p['z']:.2f} "
                  f"Rx={p['rx']:.2f} Ry={p['ry']:.2f} Rz={p['rz']:.2f}")


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    # 运行测试
    test_board_detection()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华沿机器人 + OAK相机 手内标定 (Eye-in-Hand)

标定原理:
  相机固定在机械臂末端，通过机械臂运动到多个位姿，
  记录每个位姿下的 (机器人位姿, 标定板位姿)，
  用 AX=XB 求解 endTcam = 相机 → TCP_111 的变换矩阵。

关键设定:
  1. 欧拉角顺序: ZYX (Rz@Ry@Rx) — 工业机器人标准
  2. 机器人位姿: 直接用 HRIF_ReadActTcpPos 读取 TCP_111 在基座下的位姿
  3. 标定板位姿: 用 solvePnP 检测棋盘格得到 camTboard
"""

import sys
import time
import json
import cv2
import numpy as np
import depthai as dai
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

try:
    from CPS import CPSClient
    print("✅ 华沿SDK 导入成功!")
except ImportError as e:
    print(f"❌ 华沿SDK 导入失败: {e}")
    sys.exit(1)


@dataclass
class BoardPose:
    """标定板位姿 (在相机坐标系下)"""
    success: bool
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    rvec: np.ndarray = None
    tvec: np.ndarray = None
    corners: np.ndarray = None
    reprojection_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "x": self.x, "y": self.y, "z": self.z,
            "rx": self.rx, "ry": self.ry, "rz": self.rz,
            "reprojection_error": self.reprojection_error
        }


@dataclass
class CalibrationSample:
    """一组标定数据: 机器人位姿 + 标定板位姿"""
    robot_pose: Dict[str, float]       # baseTtcp111 — TCP_111 在基座下的位姿
    board_pose: Dict[str, float]       # camTboard  — 标定板在相机下的位姿
    board_rvec: np.ndarray = None
    board_tvec: np.ndarray = None
    reprojection_error: float = 0.0
    timestamp: float = 0.0


class RobotController:
    """华沿机器人控制器 (用于标定)"""

    def __init__(self, robot_ip="192.168.0.10", robot_port=10003):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.box_id = 0
        self.rbt_id = 0
        self.cps = CPSClient()

    def connect(self):
        """连接并启动机器人"""
        print("  连接控制器...")
        ret = self.cps.HRIF_Connect(self.box_id, self.robot_ip, self.robot_port)
        if ret != 0:
            return False, f"连接失败: {ret}"
        print("  ✓ 控制器已连接")

        result = []
        for i in range(30):
            ret = self.cps.HRIF_IsControllerStarted(self.box_id, result)
            if ret == 0 and len(result) > 0 and int(result[0]) == 1:
                print("  ✓ 控制器已启动")
                break
            time.sleep(0.5)
        else:
            return False, "控制器启动超时"

        print("  正在使能...")
        ret = self.cps.HRIF_GrpEnable(self.box_id, self.rbt_id)
        if ret != 0:
            return False, f"使能失败, 错误码: {ret}"
        print("  ✓ 已使能")

        return True, "连接成功"

    # ---------- 位姿读取 ----------

    def read_pose(self):
        """
        读取 TCP_111 在基座下的位姿

        直接使用 HRIF_ReadActTcpPos, 返回 [X,Y,Z,RX,RY,RZ]
        注意: 这是完整的 TCP_111 位姿 (已包含工具偏移)
        """
        result = []
        ret = self.cps.HRIF_ReadActTcpPos(self.box_id, self.rbt_id, result)
        if ret != 0 or len(result) < 6:
            return False, None

        return True, {
            "x": float(result[0]), "y": float(result[1]), "z": float(result[2]),
            "rx": float(result[3]), "ry": float(result[4]), "rz": float(result[5]),
        }

    # ---------- 运动控制 ----------

    def move_to_joint(self, joint_pos, velocity=20, acc=40):
        """移动到目标关节角 (MoveJ)"""
        ret = self.cps.HRIF_MoveJ(
            self.box_id, self.rbt_id,
            [0, 0, 0, 0, 0, 0],
            joint_pos,
            "TCP_111", "Base",
            velocity, acc, 0.0,
            1, 0, 0, 0, "0"
        )
        if ret != 0:
            return False
        return self._wait_motion_done()

    def _wait_motion_done(self, timeout=30):
        """等待运动完成"""
        start = time.time()
        result = []
        while time.time() - start < timeout:
            ret = self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
            if ret == 0 and len(result) > 0 and str(result[0]).lower() == "true":
                return True
            time.sleep(0.05)
        return False

    # ---------- 工具方法 ----------

    @staticmethod
    def _pose_to_homogeneous(pose):
        """位姿 → 4x4 齐次矩阵 (ZYX 欧拉角)"""
        T = np.eye(4)
        rx, ry, rz = np.deg2rad([pose['rx'], pose['ry'], pose['rz']])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx),  np.cos(rx)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0, 0, 1]])

        T[:3, :3] = Rz @ Ry @ Rx  # ZYX 顺序
        T[:3, 3] = [pose['x'], pose['y'], pose['z']]
        return T

    @staticmethod
    def _homogeneous_to_pose(T):
        """4x4 齐次矩阵 → 位姿 (ZYX 欧拉角)"""
        x, y, z = T[:3, 3]
        R = T[:3, :3]

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        return {
            "x": float(x), "y": float(y), "z": float(z),
            "rx": np.rad2deg(rx), "ry": np.rad2deg(ry), "rz": np.rad2deg(rz),
        }


# ============================================================
#  OAK-D 相机 + 棋盘格检测
# ============================================================
class OAKDetector:
    """OAK-D 相机标定板检测"""

    def __init__(self, camera_ip="169.254.1.222",
                 board_size=(8, 11),
                 square_size=30.0,
                 calib_file_path=None):
        self.camera_ip = camera_ip
        self.board_size = board_size          # 棋盘格内角点数 (w, h)
        self.square_size = square_size        # 棋盘格边长 (mm)
        self.calib_file_path = calib_file_path
        self.device = None
        self.pipeline = None
        self.queue = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.object_points = self._create_board_points()
        self.success_count = 0
        self.detection_count = 0

    def _create_board_points(self) -> np.ndarray:
        """
        生成棋盘格角点的 3D 坐标 (标定板坐标系下)

        原点在第一个角点, XY 沿棋盘格平面, Z=0
        """
        w, h = self.board_size
        objp = np.zeros((w * h, 3), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                objp[idx, 0] = j * self.square_size
                objp[idx, 1] = i * self.square_size
                objp[idx, 2] = 0.0
        return objp

    def setup_camera(self):
        """加载相机内参 (左目)"""
        if self.calib_file_path is None:
            possible_paths = [
                "stereo_calibration_results.npz",
                "./stereo_calibration_results.npz",
                "../stereo_calibration_results.npz",
            ]
        else:
            possible_paths = [self.calib_file_path]

        loaded = False
        for file_path in possible_paths:
            if Path(file_path).exists():
                try:
                    print(f"  📂 尝试加载标定文件: {file_path}")
                    data = np.load(file_path)
                    if 'K_l' in data and 'D_l' in data:
                        self.camera_matrix = data['K_l'].astype(np.float32)
                        self.dist_coeffs = data['D_l'].astype(np.float32)
                        loaded = True
                        print(f"  ✅ 从 {file_path} 加载左相机内参成功")
                        break
                except Exception as e:
                    print(f"  ⚠️ 加载 {file_path} 失败: {e}")
                    continue

        if not loaded:
            print("  ⚠️ 未找到标定文件，使用默认内参")
            self.camera_matrix = np.array([
                [796.0, 0.0, 682.1],
                [0.0, 794.6, 411.3],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            return False

        if self.camera_matrix.shape != (3, 3):
            print(f"  ⚠️ 相机内参矩阵形状异常")
            return False

        if self.dist_coeffs.shape == (5,):
            self.dist_coeffs = self.dist_coeffs.reshape(5, 1)
        elif self.dist_coeffs.shape == (1, 5):
            self.dist_coeffs = self.dist_coeffs.reshape(5, 1)
        elif self.dist_coeffs.shape != (5, 1):
            if self.dist_coeffs.size == 5:
                self.dist_coeffs = self.dist_coeffs.reshape(5, 1)
            else:
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        print("\n  📷 ===== 左相机内参 =====")
        print(f"     fx = {self.camera_matrix[0, 0]:.3f}")
        print(f"     fy = {self.camera_matrix[1, 1]:.3f}")
        print(f"     cx = {self.camera_matrix[0, 2]:.3f}")
        print(f"     cy = {self.camera_matrix[1, 2]:.3f}")
        print("  =========================")
        return True

    def start_camera(self):
        """启动 OAK-D 相机 (左目 640x400 已校正输出)"""
        try:
            self.pipeline = dai.Pipeline()
            left_camera = self.pipeline.create(dai.node.MonoCamera)
            left_camera.setBoardSocket(dai.CameraBoardSocket.LEFT)
            left_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            left_camera.setFps(30)

            right_camera = self.pipeline.create(dai.node.MonoCamera)
            right_camera.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            right_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right_camera.setFps(30)

            stereo = self.pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)

            left_camera.out.link(stereo.left)
            right_camera.out.link(stereo.right)

            xout_left = self.pipeline.create(dai.node.XLinkOut)
            xout_left.setStreamName("left")
            stereo.rectifiedLeft.link(xout_left.input)

            device_info = dai.DeviceInfo(self.camera_ip)
            self.device = dai.Device(self.pipeline, device_info)
            self.queue = self.device.getOutputQueue(name="left", maxSize=4, blocking=False)
            return True
        except Exception as e:
            print(f"✗ 相机启动失败: {e}")
            return False

    def get_frame(self):
        """获取一帧左目图像"""
        if self.queue is None:
            return None
        try:
            in_frame = self.queue.tryGet()
            if in_frame is not None:
                return in_frame.getCvFrame()
            return None
        except:
            return None

    def detect_and_pose(self, frame) -> Tuple[BoardPose, np.ndarray]:
        """
        检测棋盘格并解算位姿

        Returns:
            BoardPose: 标定板在相机坐标系下的位姿 (camTboard)
            display:   可视化图像
        """
        self.detection_count += 1
        display = frame.copy() if frame is not None else None
        if display is None:
            return BoardPose(success=False), None

        gray = frame
        w, h = self.board_size

        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        if not ret:
            cv2.putText(display, "❌ 标定板检测失败", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return BoardPose(success=False), display

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners_2d = corners.reshape(-1, 2)

        # solvePnP: 3D 棋盘格角点 → 2D 像素 → 得到 camTboard
        ret, rvec, tvec = cv2.solvePnP(
            self.object_points.astype(np.float32),
            corners_2d.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )

        if not ret:
            return BoardPose(success=False), display

        projected_points, _ = cv2.projectPoints(
            self.object_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        reprojection_error = np.mean(np.sqrt(np.sum(
            (corners_2d - projected_points.reshape(-1, 2)) ** 2, axis=1
        )))

        self.success_count += 1

        x, y, z = tvec.flatten()
        R, _ = cv2.Rodrigues(rvec)
        rx, ry, rz = self._R_to_euler(R)

        pose = BoardPose(
            success=True,
            x=x, y=y, z=z,
            rx=rx, ry=ry, rz=rz,
            rvec=rvec, tvec=tvec,
            corners=corners,
            reprojection_error=reprojection_error
        )

        self._draw_board(display, corners, pose)
        return pose, display

    def _draw_board(self, frame, corners, pose):
        """在画面上绘制标定板检测结果"""
        if corners is not None:
            cv2.drawChessboardCorners(frame, self.board_size, corners, True)

        if pose.success and pose.rvec is not None and pose.tvec is not None:
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                              pose.rvec, pose.tvec, self.square_size * 2)

            info = f"X={pose.x:.1f} Y={pose.y:.1f} Z={pose.z:.1f}mm"
            info += f" | Rx={pose.rx:.1f} Ry={pose.ry:.1f} Rz={pose.rz:.1f}°"
            info += f" | reproj={pose.reprojection_error:.2f}px"

            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            rate = self.success_count / max(self.detection_count, 1) * 100
            status = f"Detect: {self.detection_count} | Success: {self.success_count} | Rate: {rate:.1f}%"
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

    @staticmethod
    def _R_to_euler(R):
        """旋转矩阵 → ZYX 欧拉角 (度)"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0
        return np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)

    def stop_camera(self):
        if self.device:
            self.device.close()
        self.device = None
        self.queue = None
        self.pipeline = None
        cv2.destroyAllWindows()


# ============================================================
#  手眼标定核心
# ============================================================
class HandInEyeCalibration:
    """
    手内标定 (Eye-in-Hand)

    标定方程: AX = XB
      A = baseTtcp111₂⁻¹ @ baseTtcp111₁  (机器人运动)
      B = camTboard₂ @ camTboard₁⁻¹        (标定板运动)
      X = endTcam = camera → TCP_111       (待求解)

    固定标定板, 机器人运动到 N 个不同位姿,
    每个位姿记录 (baseTtcp111, camTboard),
    用 OpenCV 的 calibrateHandEye 求解.
    """

    def __init__(self,
                 robot_ip="192.168.0.10",
                 camera_ip="169.254.1.222",
                 board_size=(8, 11),
                 square_size=30.0,
                 calib_file_path=None):

        self.samples: List[CalibrationSample] = []
        self.endTcam = None
        self.reprojection_errors = []

        self.save_dir = Path("./calibration_result_hand_in_eye")
        self.save_dir.mkdir(exist_ok=True)

        self.robot = RobotController(robot_ip=robot_ip)
        self.camera = OAKDetector(camera_ip, board_size, square_size, calib_file_path)

        # 预置 19 组关节角, 确保位置和姿态都有足够变化
        self.joint_poses = [
            [-11.310, 11.517, -42.399, 7.673, -115.922, -3.972],
            [-24.440, 12.104, -49.720, 34.888, -114.966, -7.386],
            [2.673, 11.995, -40.769, -18.496, -117.067, -1.934],
            [-13.179, 23.095, -30.220, 14.466, -130.694, -2.727],
            [-17.910, -0.842, -58.594, 17.554, -98.934, -7.209],
            [-14.808, 24.414, -41.077, 15.047, -116.013, -4.344],
            [-19.816, 24.544, -44.075, 26.834, -115.100, -5.211],
            [-5.007, 24.489, -37.908, -6.391, -117.056, -3.879],
            [-13.361, 33.905, -29.202, 14.311, -128.386, -2.777],
            [-15.918, 15.264, -52.156, 15.489, -103.750, -5.793],
            [-29.586, 10.421, -52.156, 23.658, -109.232, -16.028],
            [-36.925, 14.315, -55.020, 42.105, -112.865, -19.222],
            [-19.804, 7.505, -51.744, 4.446, -106.281, -11.705],
            [-25.616, 22.906, -37.393, 20.386, -125.683, -13.136],
            [-33.091, 3.837, -60.812, 28.388, -100.423, -18.791],
            [2.283, 14.374, -47.762, -2.493, -114.367, -15.562],
            [-7.548, 17.843, -46.824, 17.493, -118.253, -16.817],
            [9.453, 12.341, -50.383, -15.779, -111.756, -13.588],
            [9.009, 10.632, -52.120, -14.580, -109.493, -13.561],
        ]

    def connect_devices(self):
        """连接机器人和相机"""
        success, msg = self.robot.connect()
        if not success:
            return False, msg
        print("  ✓ 机器人连接成功")

        print("  初始化OAK相机...")
        if not self.camera.setup_camera():
            print("  ⚠️ 使用默认内参（标定精度可能受影响）")
        if not self.camera.start_camera():
            return False, "相机启动失败"
        print("  ✓ 相机已启动")
        return True, "所有设备已就绪"

    def disconnect_devices(self):
        self.robot.disconnect()
        self.camera.stop_camera()

    def collect_samples(self, num_samples: int = 20) -> bool:
        """
        采集标定数据

        机器人自动运动到预置位姿, 在每个位姿下:
          1. 读取 TCP_111 位姿 (HRIF_ReadActTcpPos)
          2. 检测标定板, 获取 camTboard
          3. 保存为一组 CalibrationSample
        """
        print(f"\n{'=' * 60}")
        print(f"开始采集 {num_samples} 组标定数据")
        print("=" * 60)
        print("操作说明:")
        print("  1. 标定板固定在桌面上不动")
        print("  2. 机器人会自动移动到多个位姿")
        print("  3. 每个位姿等待2秒，让相机稳定")
        print("  4. 检测到标定板后自动记录")
        print("  5. 按 'q' 可随时退出")
        print("=" * 60)

        for i in range(num_samples):
            idx = i % len(self.joint_poses)
            target = self.joint_poses[idx]

            print(f"\n[{i + 1}/{num_samples}] 移动到位姿: "
                  f"J1={target[0]:.1f} J2={target[1]:.1f} J3={target[2]:.1f}")

            if not self.robot.move_to_joint(target, velocity=20, acc=40):
                print("  ⚠ 移动失败，跳过")
                continue

            print("  ⏳ 等待机器人稳定...")
            time.sleep(2.0)

            # 读机器人位姿 (baseTtcp111)
            success, robot_pose = self.robot.read_pose()
            if not success:
                print("  ⚠ 读取机器人位姿失败，跳过")
                continue

            # 检测标定板 (camTboard)
            pose = None
            for retry in range(3):
                frame = self.camera.get_frame()
                if frame is None:
                    print("  ⚠ 获取图像失败，重试...")
                    time.sleep(0.5)
                    continue
                pose, display = self.camera.detect_and_pose(frame)

                cv2.imshow("Board Detection", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("  ⏹ 用户退出采集")
                    return False

                if pose.success:
                    break
                else:
                    print(f"  ⚠ 标定板检测失败 (尝试 {retry + 1}/3)，等待0.5秒后重试...")
                    time.sleep(0.5)

            if not pose or not pose.success:
                print("  ❌ 标定板检测失败，跳过此位姿")
                continue

            self.samples.append(CalibrationSample(
                robot_pose=robot_pose,
                board_pose=pose.to_dict(),
                board_rvec=pose.rvec,
                board_tvec=pose.tvec,
                reprojection_error=pose.reprojection_error,
                timestamp=time.time()
            ))
            print(f"  ✅ 采集成功 [{len(self.samples)}]")
            print(f"    TCP_111: X={robot_pose['x']:.2f} Y={robot_pose['y']:.2f} Z={robot_pose['z']:.2f}")
            print(f"    标定板:  X={pose.x:.2f} Y={pose.y:.2f} Z={pose.z:.2f} (重投影误差: {pose.reprojection_error:.2f}px)")

            time.sleep(0.5)

        print(f"\n{'=' * 60}")
        print(f"采集完成: {len(self.samples)}/{num_samples} 组")
        return len(self.samples) >= 8

    # ---------- 欧拉角工具 (ZYX 顺序) ----------

    def _pose_to_homogeneous(self, pose: dict) -> np.ndarray:
        """
        位姿 → 4x4 齐次矩阵

        欧拉角顺序: ZYX (Rz @ Ry @ Rx)
        与华沿机器人一致
        """
        T = np.eye(4)
        rx, ry, rz = np.deg2rad([pose['rx'], pose['ry'], pose['rz']])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx),  np.cos(rx)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0, 0, 1]])

        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [pose['x'], pose['y'], pose['z']]
        return T

    def _homogeneous_to_pose(self, T: np.ndarray) -> dict:
        """4x4 矩阵 → 位姿 (ZYX 欧拉角)"""
        x, y, z = T[:3, 3]
        R = T[:3, :3]

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        return {
            "x": float(x), "y": float(y), "z": float(z),
            "rx": np.rad2deg(rx), "ry": np.rad2deg(ry), "rz": np.rad2deg(rz),
        }

    # ---------- 标定计算 ----------

    def _prepare_data(self):
        """
        准备标定数据

        calibrateHandEye 需要:
          - R_gripper2base, t_gripper2base: baseTtcp111 (机器人位姿)
          - R_target2cam, t_target2cam:      camTboard  (标定板位姿)

        输出: endTcam = cam2gripper = camera → TCP_111
        """
        R_gripper2base = []
        t_gripper2base = []
        R_board2cam = []
        t_board2cam = []

        for sample in self.samples:
            # 机器人位姿: baseTtcp111
            T_robot = self._pose_to_homogeneous(sample.robot_pose)
            R_gripper2base.append(T_robot[:3, :3])
            t_gripper2base.append(T_robot[:3, 3])

            # 标定板位姿: camTboard (board in camera frame)
            T_board2cam = self._pose_to_homogeneous(sample.board_pose)
            R_board2cam.append(T_board2cam[:3, :3])
            t_board2cam.append(T_board2cam[:3, 3])

        return R_gripper2base, t_gripper2base, R_board2cam, t_board2cam

    def calibrate_with_diagnosis(self) -> bool:
        """
        执行标定, 自动选择最优方向

        同时试两种 board_pose 方向:
          方法1: board_pose 直接作为 camTboard
          方法2: board_pose 取反作为 camTboard⁻¹
        选择一致性误差小的结果
        """
        if len(self.samples) < 8:
            print(f"❌ 样本不足 (需要>=8, 当前{len(self.samples)})")
            return False

        print(f"\n{'=' * 60}")
        print("执行手内标定计算 (AX=XB)")
        print("=" * 60)
        print(f"样本数: {len(self.samples)}")
        print(f"欧拉角顺序: ZYX (Rz@Ry@Rx)")

        R_gripper2base, t_gripper2base, R_board2cam, t_board2cam = self._prepare_data()

        # ====== 方法1: board_pose 直接使用 ======
        print("\n【方法1】board_pose 直接使用 (board→camera)")
        print("-" * 40)

        try:
            R_cam2gripper_1, t_cam2gripper_1 = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_board2cam, t_board2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            endTcam_1 = np.eye(4)
            endTcam_1[:3, :3] = R_cam2gripper_1
            endTcam_1[:3, 3] = t_cam2gripper_1.flatten()

            pose_1 = self._homogeneous_to_pose(endTcam_1)
            print(f"  结果: X={pose_1['x']:.1f} Y={pose_1['y']:.1f} Z={pose_1['z']:.1f}")
            print(f"        Rx={pose_1['rx']:.1f} Ry={pose_1['ry']:.1f} Rz={pose_1['rz']:.1f}")

            errors_1 = self._compute_consistency_errors(endTcam_1)
            avg_1 = np.mean(errors_1) if errors_1 else float('inf')
            print(f"  一致性误差: 平均={avg_1:.1f}mm")

        except Exception as e:
            print(f"  ❌ 方法1失败: {e}")
            endTcam_1 = None
            avg_1 = float('inf')

        # ====== 方法2: board_pose 取反 ======
        print("\n【方法2】board_pose 取反 (camera→board)")
        print("-" * 40)

        try:
            R_board2cam_inv = []
            t_board2cam_inv = []

            for sample in self.samples:
                T_board2cam = self._pose_to_homogeneous(sample.board_pose)
                T_cam2board = np.linalg.inv(T_board2cam)
                R_board2cam_inv.append(T_cam2board[:3, :3])
                t_board2cam_inv.append(T_cam2board[:3, 3])

            R_cam2gripper_2, t_cam2gripper_2 = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_board2cam_inv, t_board2cam_inv,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            endTcam_2 = np.eye(4)
            endTcam_2[:3, :3] = R_cam2gripper_2
            endTcam_2[:3, 3] = t_cam2gripper_2.flatten()

            pose_2 = self._homogeneous_to_pose(endTcam_2)
            print(f"  结果: X={pose_2['x']:.1f} Y={pose_2['y']:.1f} Z={pose_2['z']:.1f}")
            print(f"        Rx={pose_2['rx']:.1f} Ry={pose_2['ry']:.1f} Rz={pose_2['rz']:.1f}")

            errors_2 = self._compute_consistency_errors(endTcam_2)
            avg_2 = np.mean(errors_2) if errors_2 else float('inf')
            print(f"  一致性误差: 平均={avg_2:.1f}mm")

        except Exception as e:
            print(f"  ❌ 方法2失败: {e}")
            endTcam_2 = None
            avg_2 = float('inf')

        # ====== 选择最优 ======
        print("\n【选择】比较两种方法")
        print("-" * 40)
        print(f"  方法1 平均误差: {avg_1:.1f}mm")
        print(f"  方法2 平均误差: {avg_2:.1f}mm")

        if avg_1 < avg_2 and endTcam_1 is not None:
            print(f"\n  ✅ 选择方法1 (board_pose直接使用)")
            self.endTcam = endTcam_1
        elif endTcam_2 is not None:
            print(f"\n  ✅ 选择方法2 (board_pose取反)")
            self.endTcam = endTcam_2
        else:
            print("\n  ❌ 两种方法都失败了")
            return False

        min_avg = min(avg_1, avg_2)
        if min_avg > 50:
            print("\n  ❌ 误差仍然很大 (>50mm)，可能原因:")
            print("     1. 欧拉角顺序不正确")
            print("     2. 标定板检测不稳定")
            print("     3. 相机内参不准确")
            print("     4. 样本位姿变化不够多样")

        self._validate_calibration()
        return True

    # ---------- 验证 ----------

    def _compute_consistency_errors(self, endTcam) -> List[float]:
        """
        计算一致性误差

        对固定不动的标定板:
          baseTboard = baseTtcp111 @ endTcam @ camTboard
        对所有样本, baseTboard 应相同, 位置差异即为误差
        """
        if len(self.samples) < 2 or endTcam is None:
            return []

        errors = []
        ref_sample = self.samples[0]
        T_base2end_ref = self._pose_to_homogeneous(ref_sample.robot_pose)   # baseTtcp111
        T_board2cam_ref = self._pose_to_homogeneous(ref_sample.board_pose)  # camTboard
        T_base2board_ref = T_base2end_ref @ endTcam @ T_board2cam_ref

        for sample in self.samples[1:]:
            T_base2end = self._pose_to_homogeneous(sample.robot_pose)
            T_board2cam = self._pose_to_homogeneous(sample.board_pose)
            T_base2board = T_base2end @ endTcam @ T_board2cam

            error = np.linalg.norm(T_base2board[:3, 3] - T_base2board_ref[:3, 3])
            errors.append(error)

        return errors

    def _validate_calibration(self):
        """输出标定验证结果"""
        print("\n" + "-" * 40)
        print("验证标定结果")
        print("-" * 40)

        if self.endTcam is None:
            print("  ❌ 没有标定结果")
            return

        errors = self._compute_consistency_errors(self.endTcam)

        if errors:
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            std_error = np.std(errors)

            for i, err in enumerate(errors[:10]):
                print(f"  样本{i+2}一致性误差: {err:.3f} mm")
            if len(errors) > 10:
                print(f"  ... (共{len(errors)}个样本)")

            print(f"\n  平均一致性误差: {avg_error:.3f} mm")
            print(f"  最大一致性误差: {max_error:.3f} mm")
            print(f"  标准差: {std_error:.3f} mm")

            if avg_error < 3:
                print("  ✅ 标定精度优秀！")
            elif avg_error < 5:
                print("  ✅ 标定精度良好")
            elif avg_error < 10:
                print("  ⚠️ 标定精度一般，建议增加样本或检查数据质量")
            else:
                print("  ❌ 标定精度较差")

            self.reprojection_errors = errors

    # ---------- 保存 ----------

    def save_results(self):
        """保存 endTcam.npy + calibration_result.json"""
        if self.endTcam is None:
            print("没有标定结果可保存")
            return

        np.save(self.save_dir / "endTcam.npy", self.endTcam)

        pose = self._homogeneous_to_pose(self.endTcam)
        result = {
            "timestamp": datetime.now().isoformat(),
            "calibration_type": "hand_in_eye_fixed_board_auto",
            "euler_order": "ZYX (Rz@Ry@Rx)",
            "robot_pose_source": "HRIF_ReadActTcpPos (TCP_111)",
            "num_samples": len(self.samples),
            "endTcam": {
                "matrix": self.endTcam.tolist(),
                "pose": {
                    "x": pose["x"], "y": pose["y"], "z": pose["z"],
                    "rx": pose["rx"], "ry": pose["ry"], "rz": pose["rz"]
                }
            },
            "validation": {
                "avg_error_mm": float(np.mean(self.reprojection_errors)) if self.reprojection_errors else 0,
                "max_error_mm": float(np.max(self.reprojection_errors)) if self.reprojection_errors else 0,
            }
        }

        with open(self.save_dir / "calibration_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ 结果已保存到: {self.save_dir}")
        print(f"   - endTcam.npy (4x4矩阵, camera→TCP_111)")
        print(f"   - calibration_result.json (详细信息)")


# ============================================================
#  主程序
# ============================================================
def main():
    print("=" * 60)
    print("华沿机器人 + OAK相机 手内标定 (Eye-in-Hand)")
    print("=" * 60)

    ROBOT_IP = "192.168.0.10"
    CAMERA_IP = "169.254.1.222"
    BOARD_SIZE = (8, 11)
    SQUARE_SIZE = 30.0
    NUM_SAMPLES = 20
    CALIB_FILE_PATH = "stereo_calibration_results.npz"

    calibrator = HandInEyeCalibration(
        robot_ip=ROBOT_IP,
        camera_ip=CAMERA_IP,
        board_size=BOARD_SIZE,
        square_size=SQUARE_SIZE,
        calib_file_path=CALIB_FILE_PATH
    )

    try:
        print("\n[1] 连接设备...")
        print("-" * 40)
        success, msg = calibrator.connect_devices()
        if not success:
            print(f"❌ {msg}")
            return
        print(f"✅ {msg}")

        print("\n[2] 准备就绪")
        print("-" * 40)
        print("  请确保:")
        print("  1. 标定板固定在桌面上不动")
        print("  2. OAK相机固定在机械臂末端")
        print("  3. 标定板在相机视野内")
        print("  4. 机器人已使能")
        input("\n  按 Enter 开始...")

        print("\n[3] 采集标定数据")
        if not calibrator.collect_samples(num_samples=NUM_SAMPLES):
            print("❌ 数据采集失败")
            return
        print("✅ 数据采集完成")

        print("\n[4] 执行标定计算")
        if not calibrator.calibrate_with_diagnosis():
            print("❌ 标定失败")
            return
        print("✅ 标定完成")

        print("\n[5] 保存标定结果")
        calibrator.save_results()

        print("\n" + "=" * 60)
        print("🎉 手内标定完成!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
    except Exception as e:
        print(f"\n❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        calibrator.disconnect_devices()


if __name__ == "__main__":
    main()

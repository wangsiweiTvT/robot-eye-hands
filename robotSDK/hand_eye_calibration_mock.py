#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华沿机器人 + OAK相机 手眼标定程序
支持两种模式:
  - mock: 只模拟机器人位姿（不移动真实机器人），标定板位姿由真实相机提供
  - real: 连接真实设备完整运行
"""

import sys
import os
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, List
import random

# ==================== 模式配置 ====================
# "mock": 模拟机器人位姿 + 真实相机检测标定板
# "real": 真实机器人 + 真实相机
RUN_MODE = "mock"  # 改成 "real" 使用真实机器人

# ==================== 导入SDK (Real模式下需要) ====================
if RUN_MODE == "real":
    try:
        import depthai as dai
        from CPS import CPSClient

        print("✅ SDK 导入成功!")
    except ImportError as e:
        print(f"❌ SDK 导入失败: {e}")
        sys.exit(1)
else:
    print("🔵 Mock模式: 模拟机器人位姿，使用真实相机检测标定板")


# ==================== 数据结构 ====================
@dataclass
class BoardPose:
    """标定板位姿数据"""
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
    error: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "x": self.x, "y": self.y, "z": self.z,
            "rx": self.rx, "ry": self.ry, "rz": self.rz,
            "error": self.error
        }


@dataclass
class CalibrationSample:
    """单组标定数据样本"""
    robot_pose: dict  # 法兰盘在基坐标系下的位姿
    board_pose: dict  # 标定板在相机坐标系下的位姿
    timestamp: float = 0.0


# ==================== 机器人位姿模拟器 ====================
class RobotPoseSimulator:
    """
    模拟机器人法兰盘位姿 (baseTend)
    生成在空间中均匀分布的不同位姿
    """

    def __init__(self):
        self.pose_count = 0

        # 预设的机器人位姿序列 (J1~J6 关节角度)
        # 这些姿态覆盖了不同的位置和朝向，适合手眼标定
        self.joint_poses = [
            [0, -15, 30, 0, 45, 0],
            [0, -15, 30, 0, 0, 0],
            [0, -15, 30, 0, -45, 0],
            [15, -15, 30, 0, 45, 0],
            [15, -15, 30, 0, 0, 0],
            [15, -15, 30, 0, -45, 0],
            [-15, -15, 30, 0, 45, 0],
            [-15, -15, 30, 0, 0, 0],
            [-15, -15, 30, 0, -45, 0],
            [0, -20, 35, 10, 45, 10],
            [0, -20, 35, -10, 45, -10],
            [20, -20, 35, 10, 45, 10],
            [-20, -20, 35, -10, 45, -10],
            [10, -25, 40, 15, 30, 15],
            [-10, -25, 40, -15, 30, -15],
            [0, -30, 45, 0, 60, 0],
        ]

    def get_joint_pose(self, index: int) -> List[float]:
        """获取第 index 个关节位姿（循环使用）"""
        idx = index % len(self.joint_poses)
        base = self.joint_poses[idx]

        # 添加少量随机变化，让姿态更丰富
        variation = 3
        return [
            base[0] + random.uniform(-variation, variation),
            base[1] + random.uniform(-variation, variation),
            base[2] + random.uniform(-variation, variation),
            base[3] + random.uniform(-variation, variation),
            base[4] + random.uniform(-variation, variation),
            base[5] + random.uniform(-variation, variation),
        ]

    def get_pose_count(self) -> int:
        return len(self.joint_poses)


# ==================== 真实OAK相机 (Mock和Real共用) ====================
class RealOAKDetector:
    """OAK相机标定板检测器"""

    def __init__(self, camera_ip="169.254.1.222",
                 board_size=(8, 11),
                 square_size=30.0):
        self.camera_ip = camera_ip
        self.board_size = board_size
        self.square_size = square_size
        self.device = None
        self.pipeline = None
        self.queue = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.object_points = self._create_board_points()
        self.detection_count = 0
        self.success_count = 0

    def _create_board_points(self) -> np.ndarray:
        w, h = self.board_size
        objp = np.zeros((w * h, 3), dtype=np.float32)
        offset_x = (w - 1) * self.square_size / 2.0
        offset_y = (h - 1) * self.square_size / 2.0
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                objp[idx, 0] = j * self.square_size - offset_x
                objp[idx, 1] = i * self.square_size - offset_y
                objp[idx, 2] = 0.0
        return objp

    def setup_camera(self):
        """设置相机内参（使用默认值）"""
        self.camera_matrix = np.array([
            [600.0, 0.0, 640.0],
            [0.0, 600.0, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return True

    def start_camera(self):
        """启动OAK相机"""
        try:
            import depthai as dai

            self.pipeline = dai.Pipeline()
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(640, 480)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)

            device_info = dai.DeviceInfo(self.camera_ip)
            self.device = dai.Device(self.pipeline, device_info)
            self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            return True
        except Exception as e:
            print(f"❌ 相机启动失败: {e}")
            return False

    def get_frame(self):
        """获取一帧图像"""
        if self.queue is None:
            return None
        try:
            in_frame = self.queue.tryGet()
            if in_frame is not None:
                return in_frame.getCvFrame()
            return None
        except:
            return None

    def detect_and_pose(self, frame) -> BoardPose:
        """检测标定板并计算位姿"""
        self.detection_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w, h = self.board_size
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        if not ret:
            return BoardPose(success=False)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        corners_2d = corners.reshape(-1, 2)
        ret, rvec, tvec = cv2.solvePnP(
            self.object_points.astype(np.float32),
            corners_2d.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ret:
            return BoardPose(success=False)

        x, y, z = tvec.flatten() * 1.0
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rx, ry, rz = self._rotation_matrix_to_euler(rotation_matrix)

        self.success_count += 1
        return BoardPose(
            success=True,
            x=x, y=y, z=z,
            rx=rx, ry=ry, rz=rz,
            rvec=rvec,
            tvec=tvec,
            corners=corners,
            error=0.0
        )

    def _rotation_matrix_to_euler(self, R):
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

    def stop_camera(self):
        if self.device:
            self.device.close()
            self.device = None
        self.queue = None
        self.pipeline = None


# ==================== 真实机器人控制器 ====================
class RealRobotController:
    """华沿机器人控制器"""

    def __init__(self, robot_ip="192.168.0.10", robot_port=10003):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.box_id = 0
        self.rbt_id = 0
        self.cps = CPSClient()
        self.is_connected = False

    def connect(self):
        ret = self.cps.HRIF_Connect(self.box_id, self.robot_ip, self.robot_port)
        if ret != 0:
            return False, f"连接失败: {ret}"

        result = []
        for i in range(30):
            ret = self.cps.HRIF_IsControllerStarted(self.box_id, result)
            if ret == 0 and len(result) > 0 and int(result[0]) == 1:
                break
            time.sleep(0.5)
        else:
            return False, "控制器启动超时"

        ret = self.cps.HRIF_GrpEnable(self.box_id, self.rbt_id)
        if ret != 0:
            return False, f"使能失败: {ret}"

        self.is_connected = True
        return True, "连接成功"

    def disconnect(self):
        try:
            self.cps.HRIF_DisConnect(self.box_id)
        except:
            pass
        self.is_connected = False

    def read_pose(self):
        """读取法兰盘位姿 (baseTend)"""
        result = []
        ret = self.cps.HRIF_ReadActTcpPos(self.box_id, self.rbt_id, result)
        if ret != 0 or len(result) < 6:
            return False, None
        return True, {
            "x": float(result[0]),
            "y": float(result[1]),
            "z": float(result[2]),
            "rx": float(result[3]),
            "ry": float(result[4]),
            "rz": float(result[5]),
        }

    def move_to_joint(self, joint_pos, velocity=30, acc=60):
        """移动到目标关节位置"""
        ret = self.cps.HRIF_MoveJ(
            self.box_id, self.rbt_id,
            [0, 0, 0, 0, 0, 0],
            joint_pos,
            "TCP", "Base",
            velocity, acc, 0.0,
            1, 0, 0, 0, "0"
        )
        if ret != 0:
            return False
        return self._wait_motion_done()

    def _wait_motion_done(self, timeout=30):
        start = time.time()
        result = []
        while time.time() - start < timeout:
            ret = self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
            if ret == 0 and len(result) > 0 and str(result[0]).lower() == "true":
                return True
            time.sleep(0.05)
        return False


# ==================== 手眼标定器 ====================
class HandEyeCalibration:
    """手眼标定主类"""

    def __init__(self,
                 mode="mock",
                 robot_ip="192.168.0.10",
                 camera_ip="169.254.1.222",
                 board_type="chessboard",
                 board_size=(8, 11),
                 square_size=30.0,
                 endTboard_offset=(0, 0, 5, 0, 0, 0)):
        """
        初始化

        Args:
            mode: "mock" 或 "real"
            robot_ip: 机器人IP
            camera_ip: OAK相机IP
            board_type: 标定板类型
            board_size: 棋盘格内角点数
            square_size: 方格大小 [mm]
            endTboard_offset: 标定板在法兰盘坐标系下的偏移 (X, Y, Z, Rx, Ry, Rz)
        """
        self.mode = mode
        self.samples: List[CalibrationSample] = []
        self.baseTcamera = None
        self.reprojection_errors = []

        # endTboard (标定板在法兰盘坐标系下的偏移)
        self.endTboard_offset = endTboard_offset

        self.save_dir = Path("./calibration_result")
        self.save_dir.mkdir(exist_ok=True)

        # 初始化设备
        if mode == "mock":
            print("🔵 Mock模式: 模拟机器人位姿，使用真实相机检测标定板")
            self.robot = None
            self.pose_simulator = RobotPoseSimulator()
        else:
            print("🔴 Real模式: 连接真实机器人")
            self.robot = RealRobotController(robot_ip=robot_ip)
            self.pose_simulator = None

        # 相机 (两种模式都用真实相机)
        self.camera = RealOAKDetector(
            camera_ip=camera_ip,
            board_size=board_size,
            square_size=square_size
        )

    # ==================== 连接设备 ====================

    def connect_devices(self) -> Tuple[bool, str]:
        """连接所有设备"""
        # 连接相机 (两种模式都需要)
        print("  初始化OAK相机...")
        self.camera.setup_camera()
        if not self.camera.start_camera():
            return False, "相机启动失败"
        print("  ✅ 相机已启动")

        # 如果是Real模式，连接机器人
        if self.mode == "real":
            print("  连接机器人...")
            success, msg = self.robot.connect()
            if not success:
                return False, msg
            print("  ✅ 机器人已连接")
        else:
            print("  🔵 Mock模式: 跳过机器人连接")

        return True, "所有设备已就绪"

    def disconnect_devices(self):
        """断开所有设备"""
        if self.mode == "real" and self.robot:
            self.robot.disconnect()
        if self.camera:
            self.camera.stop_camera()
        cv2.destroyAllWindows()

    # ==================== 采集数据 ====================

    def capture_sample(self, joint_target: List[float] = None) -> Tuple[bool, str]:
        """
        采集一组标定数据

        Args:
            joint_target: 目标关节位置 (Mock模式: 传入位姿索引; Real模式: 传入目标关节角度)
        """
        # 1. 获取机器人位姿
        if self.mode == "mock":
            # Mock模式: 直接生成模拟的机器人位姿
            if joint_target is None:
                joint_target = self.pose_simulator.get_joint_pose(len(self.samples))
            robot_pose = {
                "x": 350 + 150 * np.sin(len(self.samples) * 0.5 + 0.2),
                "y": 50 + 100 * np.sin(len(self.samples) * 0.7 + 1.3),
                "z": 350 + 100 * np.cos(len(self.samples) * 0.3 + 0.5),
                "rx": 20 * np.sin(len(self.samples) * 0.4 + 0.2),
                "ry": 30 * np.cos(len(self.samples) * 0.6 + 1.5),
                "rz": 15 * np.sin(len(self.samples) * 0.5 + 0.8),
            }
            # 添加少量噪声
            noise_scale = 0.5
            robot_pose = {k: v + random.uniform(-noise_scale, noise_scale)
                          for k, v in robot_pose.items()}
        else:
            # Real模式: 移动机器人并读取位姿
            if joint_target:
                if not self.robot.move_to_joint(joint_target):
                    return False, "机器人移动失败"
                time.sleep(0.5)

            success, robot_pose = self.robot.read_pose()
            if not success:
                return False, "读取机器人位姿失败"

        # 2. 获取标定板位姿 (两种模式都用真实相机)
        frame = self.camera.get_frame()
        if frame is None:
            return False, "获取相机图像失败"

        pose = self.camera.detect_and_pose(frame)

        # 显示实时画面
        cv2.imshow("Board Detection", frame)
        cv2.waitKey(1)

        if not pose.success:
            return False, "标定板检测失败"

        board_pose = pose.to_dict()

        # 3. 保存数据
        self.samples.append(CalibrationSample(
            robot_pose=robot_pose,
            board_pose=board_pose,
            timestamp=time.time()
        ))

        return True, "采集成功"

    def collect_samples(self, num_samples: int = 15) -> bool:
        """采集多组标定数据"""
        print(f"\n开始采集 {num_samples} 组数据...")
        print("=" * 60)

        for i in range(num_samples):
            if self.mode == "mock":
                # Mock模式: 使用模拟位姿，循环使用预设姿态
                print(f"\n[{i + 1}/{num_samples}] 生成模拟机器人位姿...")
                success, msg = self.capture_sample()
            else:
                # Real模式: 生成目标关节位置并移动
                joint_poses = [
                    [0, -15, 30, 0, 45, 0],
                    [0, -15, 30, 0, 0, 0],
                    [0, -15, 30, 0, -45, 0],
                    [15, -15, 30, 0, 45, 0],
                    [15, -15, 30, 0, 0, 0],
                    [15, -15, 30, 0, -45, 0],
                    [-15, -15, 30, 0, 45, 0],
                    [-15, -15, 30, 0, 0, 0],
                    [-15, -15, 30, 0, -45, 0],
                    [0, -20, 35, 10, 45, 10],
                    [0, -20, 35, -10, 45, -10],
                    [20, -20, 35, 10, 45, 10],
                    [-20, -20, 35, -10, 45, -10],
                    [10, -25, 40, 15, 30, 15],
                    [-10, -25, 40, -15, 30, -15],
                ]
                idx = i % len(joint_poses)
                target = joint_poses[idx]
                print(f"\n[{i + 1}/{num_samples}] 移动到目标位姿...")
                print(f"  J1={target[0]:.1f} J2={target[1]:.1f} J3={target[2]:.1f} "
                      f"J4={target[3]:.1f} J5={target[4]:.1f} J6={target[5]:.1f}")
                success, msg = self.capture_sample(target)

            if success:
                print(f"  ✅ {msg}")
                print(f"    机器人: X={self.samples[-1].robot_pose['x']:.2f} "
                      f"Y={self.samples[-1].robot_pose['y']:.2f} "
                      f"Z={self.samples[-1].robot_pose['z']:.2f}")
                print(f"    标定板: X={self.samples[-1].board_pose['x']:.2f} "
                      f"Y={self.samples[-1].board_pose['y']:.2f} "
                      f"Z={self.samples[-1].board_pose['z']:.2f}")
            else:
                print(f"  ❌ {msg}")

        print(f"\n采集完成: {len(self.samples)}/{num_samples} 组")
        return len(self.samples) >= 8

    # ==================== 标定计算 ====================

    def _pose_to_homogeneous(self, pose: dict) -> np.ndarray:
        """位姿转4x4齐次矩阵"""
        T = np.eye(4)
        rx, ry, rz = np.deg2rad([pose['rx'], pose['ry'], pose['rz']])

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [pose['x'], pose['y'], pose['z']]
        return T

    def _homogeneous_to_pose(self, T: np.ndarray) -> dict:
        """4x4齐次矩阵转位姿"""
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
            "x": x, "y": y, "z": z,
            "rx": np.rad2deg(rx),
            "ry": np.rad2deg(ry),
            "rz": np.rad2deg(rz),
        }

    def calibrate(self) -> bool:
        """执行手眼标定"""
        if len(self.samples) < 8:
            print(f"样本不足 (需要>=8, 当前{len(self.samples)})")
            return False

        print(f"\n开始标定计算，使用 {len(self.samples)} 组数据...")
        print("=" * 60)

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for sample in self.samples:
            T = self._pose_to_homogeneous(sample.robot_pose)
            R_gripper2base.append(T[:3, :3])
            t_gripper2base.append(T[:3, 3])

            T_cam2board = self._pose_to_homogeneous(sample.board_pose)
            T_board2cam = np.linalg.inv(T_cam2board)
            R_target2cam.append(T_board2cam[:3, :3])
            t_target2cam.append(T_board2cam[:3, 3])

        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        self.baseTcamera = np.eye(4)
        self.baseTcamera[:3, :3] = R_cam2base
        self.baseTcamera[:3, 3] = t_cam2base.flatten()

        print("\n✅ 标定计算完成!")
        print("\n计算得到的 baseTcamera:")
        print(self.baseTcamera)

        pose = self._homogeneous_to_pose(self.baseTcamera)
        print(f"\n位姿形式:")
        print(f"  X={pose['x']:.3f} Y={pose['y']:.3f} Z={pose['z']:.3f}")
        print(f"  Rx={pose['rx']:.3f} Ry={pose['ry']:.3f} Rz={pose['rz']:.3f}")

        self._validate_calibration()
        return True

    def _validate_calibration(self):
        """验证标定结果"""
        print("\n验证标定结果...")
        errors = []
        for sample in self.samples:
            T_robot = self._pose_to_homogeneous(sample.robot_pose)
            T_cam2board = self._pose_to_homogeneous(sample.board_pose)
            T_pred = self.baseTcamera @ T_cam2board
            error = np.linalg.norm(T_robot[:3, 3] - T_pred[:3, 3])
            errors.append(error)

        avg_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"  平均平移误差: {avg_error:.3f} mm")
        print(f"  最大平移误差: {max_error:.3f} mm")
        self.reprojection_errors = errors

    # ==================== 保存结果 ====================

    def save_results(self):
        if self.baseTcamera is None:
            print("没有标定结果可保存")
            return

        np.save(self.save_dir / "baseTcamera.npy", self.baseTcamera)

        pose = self._homogeneous_to_pose(self.baseTcamera)
        result = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "num_samples": len(self.samples),
            "endTboard_offset": list(self.endTboard_offset),
            "baseTcamera": {
                "matrix": self.baseTcamera.tolist(),
                "pose": {
                    "x": pose["x"], "y": pose["y"], "z": pose["z"],
                    "rx": pose["rx"], "ry": pose["ry"], "rz": pose["rz"]
                }
            },
            "avg_error": float(np.mean(self.reprojection_errors)) if self.reprojection_errors else 0
        }

        with open(self.save_dir / "calibration_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ 结果已保存到: {self.save_dir}")


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("华沿机器人 + OAK相机 手眼标定")
    print(f"运行模式: {'🔵 MOCK (模拟机器人位姿)' if RUN_MODE == 'mock' else '🔴 REAL (真实机器人)'}")
    print("=" * 60)

    # endTboard: 标定板在法兰盘坐标系下的偏移
    # 你的标定板: 中心对齐，厚度 5mm
    END_TBOARD = (0, 0, 5, 0, 0, 0)  # X, Y, Z, Rx, Ry, Rz

    calibrator = HandEyeCalibration(
        mode=RUN_MODE,
        robot_ip="192.168.0.10",
        camera_ip="169.254.1.222",
        board_type="chessboard",
        board_size=(8, 11),
        square_size=30.0,
        endTboard_offset=END_TBOARD
    )

    try:
        # 1. 连接设备
        print("\n[1] 连接设备...")
        success, msg = calibrator.connect_devices()
        if not success:
            print(f"❌ {msg}")
            return
        print(f"✅ {msg}")

        # 2. 用户确认
        if RUN_MODE == "mock":
            print("\n🔵 Mock模式: 模拟机器人位姿，使用真实相机检测标定板")
            print("  请确保标定板在相机视野内")
            input("  按 Enter 开始采集...")
        else:
            print("\n  请确保:")
            print("  - 标定板已固定在机器人法兰盘上")
            print("  - 标定板在相机视野内")
            input("\n  按 Enter 开始采集...")

        # 3. 采集数据
        print("\n[2] 采集标定数据...")
        if not calibrator.collect_samples(num_samples=4):
            print("❌ 数据采集失败")
            return
        print("✅ 数据采集完成")

        # 4. 执行标定
        print("\n[3] 执行标定计算...")
        if not calibrator.calibrate():
            print("❌ 标定失败")
            return
        print("✅ 标定完成")

        # 5. 保存结果
        print("\n[4] 保存标定结果...")
        calibrator.save_results()

        print("\n" + "=" * 60)
        print("🎉 手眼标定完成!")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        calibrator.disconnect_devices()


if __name__ == "__main__":
    main()
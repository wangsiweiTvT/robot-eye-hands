#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华沿机器人 + OAK相机 手眼标定程序 (眼在手外)
融合: 机械臂法兰盘位姿 + OAK相机标定板位姿
"""

import sys
import os
import time
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, List
# ==================== 导入模块 ====================
try:
    from CPS import CPSClient

    print("✅ 华沿SDK 导入成功!")
except ImportError as e:
    print(f"❌ 华沿SDK 导入失败: {e}")
    sys.exit(1)


# ==================== 数据结构 ====================
@dataclass
class CalibrationSample:
    """单组标定数据样本"""
    robot_pose: dict  # 法兰盘在基坐标系下的位姿 (baseTend) {x, y, z, rx, ry, rz}
    board_pose: dict  # 标定板在相机坐标系下的位姿 (cameraTboard) {x, y, z, rx, ry, rz}
    timestamp: float = 0.0


# ==================== 手眼标定器 ====================
class HandEyeCalibration:
    """手眼标定主类"""

    def __init__(self, robot_ip="192.168.0.10", robot_port=10003):
        # 机器人
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.box_id = 0
        self.rbt_id = 0
        self.cps = CPSClient()
        self.is_connected = False

        # OAK相机
        self.board_detector = None

        # 标定数据
        self.samples: List[CalibrationSample] = []

        # 标定结果
        self.baseTcamera = None  # 4x4 变换矩阵
        self.reprojection_errors = []

        # 保存目录
        self.save_dir = Path("./calibration_result")
        self.save_dir.mkdir(exist_ok=True)

    # ==================== 机器人控制 ====================

    def connect_robot(self) -> Tuple[bool, str]:
        """连接机器人"""
        print("  连接机器人...")
        ret = self.cps.HRIF_Connect(self.box_id, self.robot_ip, self.robot_port)
        if ret != 0:
            return False, f"连接失败, 错误码: {ret}"
        print("  ✓ 控制器已连接")

        # 等待启动
        result = []
        for i in range(30):
            ret = self.cps.HRIF_IsControllerStarted(self.box_id, result)
            if ret == 0 and len(result) > 0 and int(result[0]) == 1:
                print("  ✓ 控制器已启动")
                break
            time.sleep(0.5)
        else:
            return False, "控制器启动超时"

        # 使能 (自动完成上电)
        print("  正在使能...")
        ret = self.cps.HRIF_GrpEnable(self.box_id, self.rbt_id)
        if ret != 0:
            return False, f"使能失败, 错误码: {ret}"
        print("  ✓ 已上电，已使能")

        self.is_connected = True
        return True, "连接成功"

    def disconnect_robot(self):
        """断开机器人"""
        try:
            self.cps.HRIF_DisConnect(self.box_id)
            print("  ✓ 已断开机器人连接")
        except:
            pass
        self.is_connected = False

    def read_robot_pose(self) -> Tuple[bool, dict]:
        """
        读取法兰盘在基坐标系下的位姿 (baseTend)
        这是手眼标定需要的核心数据
        """
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

    def move_to_joint(self, joint_pos: List[float], velocity: float = 30.0, acc: float = 60.0) -> bool:
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

    def _wait_motion_done(self, timeout: float = 30.0) -> bool:
        """等待运动完成"""
        start = time.time()
        result = []
        while time.time() - start < timeout:
            ret = self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
            if ret == 0 and len(result) > 0 and str(result[0]).lower() == "true":
                return True
            time.sleep(0.05)
        return False

    # ==================== OAK相机 ====================

    def init_camera(self):
        """初始化OAK相机"""
        print("  初始化OAK相机...")
        self.board_detector = OAKBoardDetector(
            board_type="chessboard",
            board_size=(7, 10),
            square_size=30.0,
            preview_size=(640, 480)
        )
        self.board_detector.setup_camera()
        if not self.board_detector.start_camera():
            return False, "相机启动失败"
        print("  ✓ OAK相机已启动")
        return True, "成功"

    def capture_board_pose(self) -> Tuple[bool, dict]:
        """
        检测标定板并获取位姿
        """
        if self.board_detector is None:
            return False, None

        # 获取图像
        frame = self.board_detector.get_frame()
        if frame is None:
            return False, None

        # 检测并计算位姿
        pose = self.board_detector.detect_and_pose(frame)

        # 显示图像（方便观察）
        cv2.imshow("Board Detection", frame)
        cv2.waitKey(1)

        if not pose.success:
            return False, None

        return True, {
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "rx": pose.rx,
            "ry": pose.ry,
            "rz": pose.rz,
        }

    # ==================== 生成标定位姿 ====================

    def generate_calibration_poses(self, num_poses: int = 15) -> List[List[float]]:
        """
        生成用于标定的机器人关节位姿
        标定板在法兰盘上，所以让机器人带着标定板在相机前摆出各种姿态
        """
        # 基础姿态 (在相机视野内)
        base_poses = [
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
        ]

        # 如果需要的数量超过基础姿态，重复并微调
        poses = []
        for i in range(num_poses):
            idx = i % len(base_poses)
            base = base_poses[idx]
            # 加一点随机变化
            import random
            variation = 5
            poses.append([
                base[0] + random.uniform(-variation, variation),
                base[1] + random.uniform(-variation, variation),
                base[2] + random.uniform(-variation, variation),
                base[3] + random.uniform(-variation, variation),
                base[4] + random.uniform(-variation, variation),
                base[5] + random.uniform(-variation, variation),
            ])

        return poses

    # ==================== 采集数据 ====================

    def collect_samples(self, num_samples: int = 15) -> bool:
        """
        采集手眼标定数据
        """
        print(f"\n开始采集标定数据，需要 {num_samples} 组...")
        print("=" * 60)

        # 生成目标位姿
        target_poses = self.generate_calibration_poses(num_samples)

        for i, target in enumerate(target_poses):
            print(f"\n[{i + 1}/{num_samples}] 移动到目标位姿...")
            print(f"  J1={target[0]:.1f} J2={target[1]:.1f} J3={target[2]:.1f} "
                  f"J4={target[3]:.1f} J5={target[4]:.1f} J6={target[5]:.1f}")

            # 1. 移动机器人
            if not self.move_to_joint(target, velocity=20, acc=40):
                print("  ⚠ 移动失败，跳过此点")
                continue

            time.sleep(0.5)

            # 2. 读取机器人法兰盘位姿
            ok, robot_pose = self.read_robot_pose()
            if not ok:
                print("  ⚠ 读取机器人位姿失败，跳过此点")
                continue

            # 3. 读取标定板位姿 (OAK相机)
            ok, board_pose = self.capture_board_pose()
            if not ok:
                print("  ⚠ 识别标定板失败，跳过此点")
                continue

            # 4. 保存数据
            sample = CalibrationSample(
                robot_pose=robot_pose,
                board_pose=board_pose,
                timestamp=time.time()
            )
            self.samples.append(sample)

            print(f"  ✅ 采集成功 [{len(self.samples)}/{num_samples}]")
            print(f"    机器人: X={robot_pose['x']:.2f} Y={robot_pose['y']:.2f} Z={robot_pose['z']:.2f}")
            print(f"    标定板: X={board_pose['x']:.2f} Y={board_pose['y']:.2f} Z={board_pose['z']:.2f}")

        print("\n" + "=" * 60)
        print(f"采集完成！成功采集 {len(self.samples)} 组数据")
        return len(self.samples) >= 8

    # ==================== 标定计算 ====================

    def _pose_to_homogeneous(self, pose: dict) -> np.ndarray:
        """将位姿转换为4x4齐次变换矩阵"""
        T = np.eye(4)

        # 欧拉角转旋转矩阵 (Z-Y-X顺序)
        rx, ry, rz = np.deg2rad([pose['rx'], pose['ry'], pose['rz']])

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [pose['x'], pose['y'], pose['z']]

        return T

    def calibrate(self) -> bool:
        """
        执行手眼标定
        """
        if len(self.samples) < 8:
            print(f"样本数量不足 (需要至少8组，当前{len(self.samples)}组)")
            return False

        print(f"\n开始手眼标定计算，使用 {len(self.samples)} 组数据...")
        print("=" * 60)

        # 准备数据
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for sample in self.samples:
            # 机器人法兰盘位姿 (baseTend)
            T = self._pose_to_homogeneous(sample.robot_pose)
            R_gripper2base.append(T[:3, :3])
            t_gripper2base.append(T[:3, 3])

            # 标定板在相机下的位姿 (cameraTboard)
            # 需要转换为 board2cam (标定板到相机)
            T_cam2board = self._pose_to_homogeneous(sample.board_pose)
            T_board2cam = np.linalg.inv(T_cam2board)
            R_target2cam.append(T_board2cam[:3, :3])
            t_target2cam.append(T_board2cam[:3, 3])

        # 使用OpenCV手眼标定
        print("  使用OpenCV进行手眼标定...")
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # 构建齐次变换矩阵
        self.baseTcamera = np.eye(4)
        self.baseTcamera[:3, :3] = R_cam2base
        self.baseTcamera[:3, 3] = t_cam2base.flatten()

        print("\n✅ 标定计算完成！")
        print("\n相机到基座的变换矩阵 (baseTcamera):")
        print(self.baseTcamera)

        # 转换为位姿显示
        pose = self._homogeneous_to_pose(self.baseTcamera)
        print(f"\n转换为位姿:")
        print(f"  X={pose['x']:.3f} mm  Y={pose['y']:.3f} mm  Z={pose['z']:.3f} mm")
        print(f"  Rx={pose['rx']:.3f}°  Ry={pose['ry']:.3f}°  Rz={pose['rz']:.3f}°")

        # 验证
        self._validate_calibration()

        return True

    def _homogeneous_to_pose(self, T: np.ndarray) -> dict:
        """将4x4齐次变换矩阵转换为位姿"""
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

    def _validate_calibration(self):
        """验证标定结果"""
        print("\n验证标定结果...")

        errors = []
        for sample in self.samples:
            # 真实: 通过机器人得到的标定板位姿
            T_robot = self._pose_to_homogeneous(sample.robot_pose)

            # 预测: 通过标定结果计算
            T_cam2board = self._pose_to_homogeneous(sample.board_pose)
            T_pred = self.baseTcamera @ T_cam2board

            # 计算误差
            error = np.linalg.norm(T_robot[:3, 3] - T_pred[:3, 3])
            errors.append(error)

        avg_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)

        print(f"  平均平移误差: {avg_error:.3f} mm")
        print(f"  最大平移误差: {max_error:.3f} mm")
        print(f"  标准差: {std_error:.3f} mm")

        self.reprojection_errors = errors

    # ==================== 保存和加载 ====================

    def save_results(self):
        """保存标定结果"""
        if self.baseTcamera is None:
            print("没有标定结果可保存")
            return

        # 保存矩阵
        np.save(self.save_dir / "baseTcamera.npy", self.baseTcamera)

        # 保存JSON
        pose = self._homogeneous_to_pose(self.baseTcamera)
        result = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self.samples),
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

        print(f"\n✅ 标定结果已保存到: {self.save_dir}")

    def load_results(self) -> bool:
        """加载标定结果"""
        matrix_path = self.save_dir / "baseTcamera.npy"
        if not matrix_path.exists():
            print("未找到标定结果文件")
            return False

        self.baseTcamera = np.load(matrix_path)
        print(f"✅ 已加载标定结果")
        return True

    # ==================== 坐标转换 ====================

    def camera_to_base(self, camera_pose: dict) -> dict:
        """
        将相机坐标系下的位姿转换到基坐标系
        """
        if self.baseTcamera is None:
            raise ValueError("未加载标定结果")

        T_cam2obj = self._pose_to_homogeneous(camera_pose)
        T_base2obj = self.baseTcamera @ T_cam2obj
        return self._homogeneous_to_pose(T_base2obj)

    # ==================== 清理 ====================

    def close(self):
        """关闭所有资源"""
        self.disconnect_robot()
        if self.board_detector:
            self.board_detector.stop_camera()
        cv2.destroyAllWindows()


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("华沿机器人 + OAK相机 手眼标定")
    print("构型: 眼在手外 (Eye-to-Hand)")
    print("=" * 60)

    # 创建标定器
    calibrator = HandEyeCalibration(robot_ip="192.168.0.10")

    try:
        # 1. 连接机器人
        print("\n[1] 连接机器人...")
        success, msg = calibrator.connect_robot()
        if not success:
            print(f"❌ {msg}")
            return
        print("✅ 机器人连接成功")

        # 2. 初始化OAK相机
        print("\n[2] 初始化OAK相机...")
        success, msg = calibrator.init_camera()
        if not success:
            print(f"❌ {msg}")
            return
        print("✅ OAK相机初始化成功")

        # 3. 用户确认
        print("\n[3] 准备采集数据...")
        print("  请确保:")
        print("  - 标定板已固定在机器人法兰盘上")
        print("  - 标定板在OAK相机视野内")
        print("  - 标定板清晰可见")
        input("\n  按 Enter 键开始采集...")

        # 4. 采集数据
        print("\n[4] 采集标定数据...")
        if not calibrator.collect_samples(num_samples=15):
            print("❌ 数据采集失败")
            return
        print("✅ 数据采集完成")

        # 5. 执行标定
        print("\n[5] 执行手眼标定...")
        if not calibrator.calibrate():
            print("❌ 标定失败")
            return
        print("✅ 标定完成")

        # 6. 保存结果
        print("\n[6] 保存标定结果...")
        calibrator.save_results()

        # 7. 测试坐标转换
        print("\n[7] 测试坐标转换...")
        test_cam_pose = {"x": 100, "y": 50, "z": 200, "rx": 0, "ry": 0, "rz": 0}
        base_pose = calibrator.camera_to_base(test_cam_pose)
        print(f"  相机坐标下的点: ({test_cam_pose['x']}, {test_cam_pose['y']}, {test_cam_pose['z']})")
        print(f"  基坐标下的点: ({base_pose['x']:.2f}, {base_pose['y']:.2f}, {base_pose['z']:.2f})")

        print("\n" + "=" * 60)
        print("🎉 手眼标定完成！")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        calibrator.close()


if __name__ == "__main__":
    main()
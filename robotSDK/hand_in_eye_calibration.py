#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华沿机器人 + OAK相机 手内标定 (Eye-in-Hand) - 全自动方案
方案: 标定板固定不动，机器人（相机）移动到多个位姿
不需要任何手动输入，全自动完成
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
from typing import Tuple, Optional, List

# ==================== 导入华沿SDK ====================
try:
    from CPS import CPSClient

    print("✅ 华沿SDK 导入成功!")
except ImportError as e:
    print(f"❌ 华沿SDK 导入失败: {e}")
    sys.exit(1)


# ==================== 数据结构 ====================
@dataclass
class BoardPose:
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

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "x": self.x, "y": self.y, "z": self.z,
            "rx": self.rx, "ry": self.ry, "rz": self.rz,
            "error": self.error
        }


@dataclass
class CalibrationSample:
    robot_pose: dict
    board_pose: dict
    timestamp: float = 0.0


# ==================== 华沿机器人控制器 ====================
class RobotController:
    def __init__(self, robot_ip="192.168.0.10", robot_port=10003):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.box_id = 0
        self.rbt_id = 0
        self.cps = CPSClient()

    def connect(self):
        """连接机器人并使用 GrpEnable 完成初始化"""
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

        # 使用 GrpEnable 自动完成电箱连接、上电、使能
        print("  正在使能...")
        ret = self.cps.HRIF_GrpEnable(self.box_id, self.rbt_id)
        if ret != 0:
            return False, f"使能失败, 错误码: {ret}"
        print("  ✓ 已使能")

        return True, "连接成功"

    def disconnect(self):
        """断开连接（不断电，只断开网络）"""
        try:
            self.cps.HRIF_DisConnect(self.box_id)
            print("  ✓ 已断开连接")
        except:
            pass

    def read_pose(self):
        """读取法兰盘位姿 (baseTend) - 笛卡尔坐标"""
        result = []
        ret = self.cps.HRIF_ReadActTcpPos(self.box_id, self.rbt_id, result)
        if ret != 0 or len(result) < 6:
            return False, None
        return True, {
            "x": float(result[0]), "y": float(result[1]), "z": float(result[2]),
            "rx": float(result[3]), "ry": float(result[4]), "rz": float(result[5]),
        }

    def move_to_joint(self, joint_pos, velocity=20, acc=40):
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


# ==================== OAK相机标定板检测器 ====================
class OAKDetector:
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
        self.camera_matrix = np.array([
            [600.0, 0.0, 640.0],
            [0.0, 600.0, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return True

    def start_camera(self):
        try:
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
            print(f"✗ 相机启动失败: {e}")
            return False

    def get_frame(self):
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

        x, y, z = tvec.flatten()
        R, _ = cv2.Rodrigues(rvec)
        rx, ry, rz = self._R_to_euler(R)

        return BoardPose(True, x, y, z, rx, ry, rz, rvec, tvec)

    def _R_to_euler(self, R):
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
        cv2.destroyAllWindows()


# ==================== 手内标定器 ====================
class HandInEyeCalibration:
    def __init__(self,
                 robot_ip="192.168.0.10",
                 camera_ip="169.254.1.222",
                 board_size=(8, 11),
                 square_size=30.0):

        self.samples: List[CalibrationSample] = []
        self.endTcam = None
        self.reprojection_errors = []

        self.save_dir = Path("./calibration_result_hand_in_eye")
        self.save_dir.mkdir(exist_ok=True)

        self.robot = RobotController(robot_ip=robot_ip)
        self.camera = OAKDetector(camera_ip, board_size, square_size)

        # 预设的机器人关节位姿（标定板固定，机器人运动）
        self.joint_poses = [
            [7.243, 6.164, -45.252, -5.300, -71.845, -17.153],
            [0.934, 6.164, -45.252, -5.300, -73.259, -13.858],
            [0.934, 11.255, -40.856, -3.611, -73.259, -13.858],
            [7.286, 11.255, -40.856, -3.611, -72.004, -13.858],
            [7.286, 4.824, -40.856, -1.697, -76.619, -13.858],
            [7.286, 4.824, -40.856, -2.803, -73.735, -1.716],
            [13.866, 16.841, -33.482, -2.803, -73.735, -1.716],
            [13.866, 27.159, -29.114, -11.451, -73.735, -1.716],
            [9.642, -4.959, -51.104, -11.451, -73.735, -1.716],
            [9.642, -17.979, -76.103, -5.869, -55.311, -2.444],
            [1.065, -17.979, -76.103, -5.869, -57.761, -2.444],
            [1.065, 21.207, -39.848, -2.140, -69.059, -2.444],
            [26.302, 21.207, -41.943, -29.758, -70.212, -2.444],
            [2.456, 7.547, -46.395, 2.592, -70.212, -2.444],
            [-14.857, 13.577, -46.395, 2.592, -70.212, 77.599],
        ]

    # ==================== 连接设备 ====================

    def connect_devices(self):
        """连接所有设备"""
        success, msg = self.robot.connect()
        if not success:
            return False, msg
        print("  ✓ 机器人连接成功")

        print("  初始化OAK相机...")
        self.camera.setup_camera()
        if not self.camera.start_camera():
            return False, "相机启动失败"
        print("  ✓ 相机已启动")

        return True, "所有设备已就绪"

    def disconnect_devices(self):
        """断开所有设备"""
        self.robot.disconnect()
        self.camera.stop_camera()

    # ==================== 数据采集 ====================

    def collect_samples(self, num_samples: int = 15) -> bool:
        """采集标定数据（全自动，无需手动输入）"""
        print(f"\n{'=' * 60}")
        print(f"开始采集 {num_samples} 组标定数据")
        print("=" * 60)
        print("操作说明:")
        print("  1. 标定板固定在桌面上不动")
        print("  2. 机器人会自动移动到多个位姿")
        print("  3. 每个位姿自动记录数据")
        print("  4. 全程不需要任何手动输入！")
        print("=" * 60)

        for i in range(num_samples):
            idx = i % len(self.joint_poses)
            target = self.joint_poses[idx]

            print(f"\n[{i + 1}/{num_samples}] 移动到位姿: "
                  f"J1={target[0]:.1f} J2={target[1]:.1f} J3={target[2]:.1f}")

            # 1. 移动机器人
            if not self.robot.move_to_joint(target, velocity=20, acc=40):
                print("  ⚠ 移动失败，跳过")
                continue
            time.sleep(0.5)

            # 2. 读取机器人位姿
            success, robot_pose = self.robot.read_pose()
            if not success:
                print("  ⚠ 读取机器人位姿失败，跳过")
                continue

            # 3. 相机检测标定板
            frame = self.camera.get_frame()
            if frame is None:
                print("  ⚠ 获取图像失败，跳过")
                continue

            pose = self.camera.detect_and_pose(frame)
            cv2.imshow("Board Detection", frame)
            cv2.waitKey(1)

            if not pose.success:
                print("  ⚠ 标定板检测失败，跳过")
                continue

            # 4. 保存样本
            self.samples.append(CalibrationSample(
                robot_pose=robot_pose,
                board_pose=pose.to_dict(),
                timestamp=time.time()
            ))
            print(f"  ✅ 采集成功 [{len(self.samples)}]")
            print(f"    机器人: X={robot_pose['x']:.2f} Y={robot_pose['y']:.2f} Z={robot_pose['z']:.2f}")
            print(f"    标定板: X={pose.x:.2f} Y={pose.y:.2f} Z={pose.z:.2f}")

        print(f"\n{'=' * 60}")
        print(f"采集完成: {len(self.samples)}/{num_samples} 组")
        return len(self.samples) >= 8

    # ==================== 坐标变换工具 ====================

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

    # ==================== 标定计算 ====================

    def calibrate(self) -> bool:
        """
        执行手内标定
        使用 AX = XB 方法 (标定板固定，机器人运动)
        cv2.calibrateHandEye 直接输出 endTcam
        """
        if len(self.samples) < 8:
            print(f"❌ 样本不足 (需要>=8, 当前{len(self.samples)})")
            return False

        print(f"\n{'=' * 60}")
        print("执行手内标定计算 (AX=XB方法)")
        print("=" * 60)
        print(f"样本数: {len(self.samples)}")
        print("=" * 60)

        # 准备数据
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for sample in self.samples:
            # 法兰盘在基座下的位姿 (baseTend)
            T = self._pose_to_homogeneous(sample.robot_pose)
            R_gripper2base.append(T[:3, :3])
            t_gripper2base.append(T[:3, 3])

            # 标定板在相机下的位姿 (camTboard)
            # 需要转换为 board2cam (标定板→相机)
            T_cam2board = self._pose_to_homogeneous(sample.board_pose)
            T_board2cam = np.linalg.inv(T_cam2board)
            R_target2cam.append(T_board2cam[:3, :3])
            t_target2cam.append(T_board2cam[:3, 3])

        # 使用OpenCV手眼标定
        # 对于"眼在手上"，输出就是 camTend (相机在法兰盘坐标系下的位姿)
        print("  使用OpenCV calibrateHandEye (TSAI方法)...")
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # 构建齐次变换矩阵: endTcam
        self.endTcam = np.eye(4)
        self.endTcam[:3, :3] = R_cam2gripper
        self.endTcam[:3, 3] = t_cam2gripper.flatten()

        print("\n" + "=" * 60)
        print("✅ 标定计算完成!")
        print("=" * 60)
        print("\n计算得到的 endTcam (相机在法兰盘坐标系下的位姿):")
        print(self.endTcam)

        pose = self._homogeneous_to_pose(self.endTcam)
        print(f"\n位姿形式:")
        print(f"  X = {pose['x']:.3f} mm")
        print(f"  Y = {pose['y']:.3f} mm")
        print(f"  Z = {pose['z']:.3f} mm")
        print(f"  Rx= {pose['rx']:.3f}°")
        print(f"  Ry= {pose['ry']:.3f}°")
        print(f"  Rz= {pose['rz']:.3f}°")

        # 验证
        self._validate_calibration()

        return True

    def _validate_calibration(self):
        """验证标定结果"""
        print("\n" + "-" * 40)
        print("验证标定结果")
        print("-" * 40)

        # 用第一组作为参考
        ref_sample = self.samples[0]
        T_base2end_ref = self._pose_to_homogeneous(ref_sample.robot_pose)
        T_cam2board_ref = self._pose_to_homogeneous(ref_sample.board_pose)
        T_base2board_ref = T_base2end_ref @ self.endTcam @ T_cam2board_ref

        errors = []
        for i, sample in enumerate(self.samples[1:], 1):
            T_base2end = self._pose_to_homogeneous(sample.robot_pose)
            T_cam2board = self._pose_to_homogeneous(sample.board_pose)
            T_base2board_pred = T_base2end @ self.endTcam @ T_cam2board

            error = np.linalg.norm(T_base2board_pred[:3, 3] - T_base2board_ref[:3, 3])
            errors.append(error)
            print(f"  样本{i + 1}一致性误差: {error:.3f} mm")

        if errors:
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            std_error = np.std(errors)
            print(f"\n  平均一致性误差: {avg_error:.3f} mm")
            print(f"  最大一致性误差: {max_error:.3f} mm")
            print(f"  标准差: {std_error:.3f} mm")

            if avg_error < 3:
                print("  ✅ 标定精度良好，可以投入使用")
            elif avg_error < 10:
                print("  ⚠ 标定精度一般，建议增加样本或检查数据质量")
            else:
                print("  ❌ 标定精度较差，建议检查标定板检测是否稳定")

            self.reprojection_errors = errors

    # ==================== 保存结果 ====================

    def save_results(self):
        if self.endTcam is None:
            print("没有标定结果可保存")
            return

        np.save(self.save_dir / "endTcam.npy", self.endTcam)

        pose = self._homogeneous_to_pose(self.endTcam)
        result = {
            "timestamp": datetime.now().isoformat(),
            "calibration_type": "hand_in_eye_fixed_board_auto",
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
        print(f"   - endTcam.npy (4x4矩阵)")
        print(f"   - calibration_result.json (详细信息)")


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("华沿机器人 + OAK相机 手内标定 (Eye-in-Hand)")
    print("方案: 标定板固定，机器人自动移动到多个位姿")
    print("全程不需要任何手动输入数据")
    print("=" * 60)

    # 配置参数
    ROBOT_IP = "192.168.0.10"
    CAMERA_IP = "169.254.1.222"
    BOARD_SIZE = (8, 11)
    SQUARE_SIZE = 30.0
    NUM_SAMPLES = 15

    calibrator = HandInEyeCalibration(
        robot_ip=ROBOT_IP,
        camera_ip=CAMERA_IP,
        board_size=BOARD_SIZE,
        square_size=SQUARE_SIZE
    )

    try:
        # 1. 连接设备
        print("\n[1] 连接设备...")
        print("-" * 40)
        success, msg = calibrator.connect_devices()
        if not success:
            print(f"❌ {msg}")
            return
        print(f"✅ {msg}")

        # 2. 用户确认
        print("\n[2] 准备就绪")
        print("-" * 40)
        print("  请确保:")
        print("  1. 标定板固定在桌面上不动")
        print("  2. OAK相机固定在法兰盘上")
        print("  3. 标定板在相机视野内")
        print("  4. 机器人已使能")
        print("\n  机器人将自动移动到多个位姿采集数据")
        print("  全程不需要手动输入任何数据")
        input("\n  按 Enter 开始...")

        # 3. 采集数据
        print("\n[3] 采集标定数据")
        if not calibrator.collect_samples(num_samples=NUM_SAMPLES):
            print("❌ 数据采集失败")
            return
        print("✅ 数据采集完成")

        # 4. 执行标定
        print("\n[4] 执行标定计算")
        if not calibrator.calibrate():
            print("❌ 标定失败")
            return
        print("✅ 标定完成")

        # 5. 保存结果
        print("\n[5] 保存标定结果")
        calibrator.save_results()

        print("\n" + "=" * 60)
        print("🎉 手内标定完成!")
        print("=" * 60)
        print("\n标定结果 endTcam 即为相机在法兰盘坐标系下的位姿")
        print("可直接设置为工具坐标使用")

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
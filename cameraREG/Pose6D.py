#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华沿机器人 + OAK相机 手眼标定程序 (眼在手外)
Eye-to-Hand Calibration
"""

import time
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

# ==================== 导入华沿SDK ====================
from CPS import CPSClient


# ==================== 数据结构 ====================
@dataclass
class Pose6D:
    """6维位姿数据: X,Y,Z [mm], Rx,Ry,Rz [deg]"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    def to_array(self) -> np.ndarray:
        return np.array(self.to_list())


@dataclass
class CalibrationSample:
    """单组标定数据样本"""
    robot_pose: Pose6D  # 机器人法兰盘在基坐标下的位姿 (baseTend)
    board_pose: Pose6D  # 标定板在相机坐标系下的位姿 (cameraP)
    timestamp: float = 0.0


# ==================== 标定数据采集 ====================
class HandEyeCalibration:
    def __init__(self, robot_ip: str = "192.168.0.10", port: int = 10003):
        """
        初始化标定器

        Args:
            robot_ip: 机器人控制器IP地址
            port: 机器人控制器端口
        """
        self.cps = CPSClient()
        self.robot_ip = robot_ip
        self.port = port
        self.box_id = 0
        self.rbt_id = 0

        # 标定数据存储
        self.samples: List[CalibrationSample] = []

        # 标定参数
        self.board_size = (7, 10)  # 棋盘格内角点数量 (宽, 高)
        self.square_size = 30.0  # 棋盘格方格大小 [mm]

        # 标定结果
        self.baseTcamera = None  # 4x4 变换矩阵
        self.reprojection_errors = []  # 重投影误差

        # 结果保存路径
        self.save_dir = Path("./calibration_result")
        self.save_dir.mkdir(exist_ok=True)

    # ==================== 机器人控制接口 ====================
    def connect_robot(self) -> bool:
        """连接机器人"""
        try:
            print(f"正在连接机器人 {self.robot_ip}:{self.port} ...")
            ret = self.cps.HRIF_Connect(self.box_id, self.robot_ip, self.port)
            if ret != 0:
                print(f"连接失败, 错误码: {ret}")
                return False

            # 等待控制器启动完成
            result = []
            for _ in range(30):
                ret = self.cps.HRIF_IsControllerStarted(self.box_id, result)
                if ret == 0 and int(result[0]) == 1:
                    print("控制器已启动")
                    break
                time.sleep(1)

            # 连接电箱并上电
            ret = self.cps.HRIF_Connect2Box(self.box_id)
            if ret != 0:
                print(f"连接电箱失败, 错误码: {ret}")
                return False

            ret = self.cps.HRIF_Electrify(self.box_id)
            if ret != 0:
                print(f"上电失败, 错误码: {ret}")
                return False

            # 使能机器人
            ret = self.cps.HRIF_GrpEnable(self.box_id, self.rbt_id)
            if ret != 0:
                print(f"使能失败, 错误码: {ret}")
                return False

            print("机器人连接并上电成功！")
            return True

        except Exception as e:
            print(f"连接机器人时发生异常: {e}")
            return False

    def disconnect_robot(self):
        """断开机器人连接"""
        try:
            self.cps.HRIF_GrpDisable(self.box_id, self.rbt_id)
            self.cps.HRIF_BlackOut(self.box_id)
            self.cps.HRIF_DisConnect(self.box_id)
            print("已断开机器人连接")
        except Exception as e:
            print(f"断开连接时发生异常: {e}")

    def move_to_pose(self, target_pose: Pose6D, velocity: float = 50.0,
                     acc: float = 100.0, radius: float = 0.0) -> bool:
        """
        移动机器人到目标位姿 (使用关节运动)

        Args:
            target_pose: 目标位姿 (关节坐标)
            velocity: 速度 [deg/s]
            acc: 加速度 [deg/s²]
            radius: 过渡半径 [mm]
        """
        try:
            joint_pos = target_pose.to_list()

            # 使用MoveJ进行关节运动
            # 参数: boxID, rbtID, 空间位置(无效), 关节位置, TCP, UCS, 速度, 加速度, 半径, 使用关节坐标
            ret = self.cps.HRIF_MoveJ(
                self.box_id, self.rbt_id,
                [0, 0, 0, 0, 0, 0],  # 空间位置 (未使用)
                joint_pos,  # 关节目标位置
                "TCP",  # 工具坐标名称
                "Base",  # 用户坐标名称
                velocity,  # 速度
                acc,  # 加速度
                radius,  # 过渡半径
                1,  # 使用关节坐标 (1=是)
                0, 0, 0,  # 不检测DI停止
                "0"  # 命令ID
            )

            if ret != 0:
                print(f"MoveJ失败, 错误码: {ret}")
                return False

            # 等待运动完成
            return self.wait_for_motion_done()

        except Exception as e:
            print(f"移动机器人时发生异常: {e}")
            return False

    def wait_for_motion_done(self, timeout: float = 30.0) -> bool:
        """等待机器人运动完成"""
        start_time = time.time()
        result = []

        while time.time() - start_time < timeout:
            ret = self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
            if ret == 0 and len(result) > 0:
                if result[0] == "True" or result[0] == True:
                    return True
            time.sleep(0.1)

        print("等待运动完成超时")
        return False

    def read_current_pose(self) -> Pose6D:
        """读取机器人当前实际关节位置"""
        result = []
        ret = self.cps.HRIF_ReadActPos(self.box_id, self.rbt_id, result)
        if ret != 0 or len(result) < 6:
            print(f"读取位置失败, 错误码: {ret}")
            return None

        return Pose6D(
            x=float(result[0]), y=float(result[1]), z=float(result[2]),
            rx=float(result[3]), ry=float(result[4]), rz=float(result[5])
        )

    # ==================== OAK相机接口 (示例) ====================
    def capture_board_pose(self) -> Tuple[bool, Pose6D]:
        """
        使用OAK相机拍摄并识别标定板位姿

        Returns:
            (success, board_pose): 标定板在相机坐标系下的位姿 (X,Y,Z [mm], Rx,Ry,Rz [deg])
        """
        # TODO: 使用OAK相机API获取图像并检测标定板
        # 这里需要替换为你实际的OAK相机识别代码
        # 参考OAK的示例: https://github.com/luxonis/depthai-python

        # 占位实现 - 实际使用时要替换为真实识别代码
        print("等待OAK相机识别标定板...")
        time.sleep(0.5)

        # 模拟标定板位姿 (实际需要从相机获取)
        board_pose = Pose6D(x=100, y=50, z=200, rx=0, ry=0, rz=0)

        return True, board_pose

    # ==================== 标定数据采集 ====================
    def generate_calibration_poses(self, num_poses: int = 20) -> List[Pose6D]:
        """
        生成用于标定的机器人位姿序列

        Args:
            num_poses: 需要生成的位姿数量

        Returns:
            位姿列表
        """
        # 基本工作位置 - 根据实际情况调整
        # 这里生成围绕一个中心点、姿态变化丰富的位姿

        poses = []

        # 基础位姿 (在机器人工作空间内)
        base_angles = [
            [0, -30, 30, 0, 45, 0],
            [0, -30, 30, 0, 0, 0],
            [0, -30, 30, 0, -45, 0],
            [30, -30, 30, 0, 45, 0],
            [30, -30, 30, 0, 0, 0],
            [30, -30, 30, 0, -45, 0],
            [-30, -30, 30, 0, 45, 0],
            [-30, -30, 30, 0, 0, 0],
            [-30, -30, 30, 0, -45, 0],
        ]

        # 生成更多变化
        variations = []
        for rx in [-20, 0, 20]:
            for ry in [-15, 0, 15]:
                for rz in [-10, 0, 10]:
                    if len(variations) < num_poses:
                        variations.append([rx, ry, rz])

        # 组合生成目标位姿
        for i in range(min(num_poses, len(base_angles) * len(variations))):
            base_idx = i % len(base_angles)
            var_idx = (i // len(base_angles)) % len(variations)

            base = base_angles[base_idx]
            var = variations[var_idx]

            pose = Pose6D(
                x=base[0] + var[0],
                y=base[1] + var[1],
                z=base[2] + var[2],
                rx=base[3] + var[0] * 0.5,
                ry=base[4] + var[1] * 0.5,
                rz=base[5] + var[2] * 0.5
            )
            poses.append(pose)

        return poses[:num_poses]

    def collect_samples(self, num_samples: int = 15) -> bool:
        """
        采集标定数据样本

        Args:
            num_samples: 采集的样本数量 (建议15-20组)
        """
        print(f"\n开始采集标定数据，共需要 {num_samples} 组...")
        print("=" * 60)

        # 生成标定位姿序列
        target_poses = self.generate_calibration_poses(num_samples)

        for i, target_pose in enumerate(target_poses):
            print(f"\n[{i + 1}/{num_samples}] 移动到目标位姿...")
            print(f"  关节位置: J1={target_pose.x:.2f}, J2={target_pose.y:.2f}, "
                  f"J3={target_pose.z:.2f}, J4={target_pose.rx:.2f}, "
                  f"J5={target_pose.ry:.2f}, J6={target_pose.rz:.2f}")

            # 移动机器人到目标位姿
            if not self.move_to_pose(target_pose, velocity=30, acc=60):
                print(f"  移动到目标位姿失败，跳过此点")
                continue

            # 等待稳定
            time.sleep(1.0)

            # 读取当前机器人位姿
            robot_pose = self.read_current_pose()
            if robot_pose is None:
                print("  读取机器人位姿失败，跳过此点")
                continue

            # 识别标定板位姿
            success, board_pose = self.capture_board_pose()
            if not success:
                print("  识别标定板失败，跳过此点")
                continue

            # 保存样本
            sample = CalibrationSample(robot_pose, board_pose, time.time())
            self.samples.append(sample)

            print(f"  ✓ 采集成功 [{len(self.samples)}/{num_samples}]")

            # 显示进度
            print(f"  机器人位姿: {robot_pose.to_list()}")
            print(f"  标定板位姿: {board_pose.to_list()}")
            print("-" * 60)

        print(f"\n采集完成！成功采集 {len(self.samples)} 组数据")
        return len(self.samples) >= 10

    # ==================== 手眼标定计算 ====================
    def pose_to_homogeneous(self, pose: Pose6D) -> np.ndarray:
        """将6维位姿转换为4x4齐次变换矩阵"""
        T = np.eye(4)

        # 旋转部分 (使用欧拉角转旋转矩阵)
        rx, ry, rz = np.deg2rad([pose.rx, pose.ry, pose.rz])

        # Rx
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        # Ry
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        # Rz
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵 (Z-Y-X顺序)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [pose.x, pose.y, pose.z]

        return T

    def homogeneous_to_pose(self, T: np.ndarray) -> Pose6D:
        """将4x4齐次变换矩阵转换为6维位姿"""
        # 提取平移
        x, y, z = T[:3, 3]

        # 提取欧拉角 (从旋转矩阵)
        R = T[:3, :3]

        # 计算欧拉角 (Z-Y-X顺序)
        if abs(R[2, 0]) < 0.9999:
            ry = -np.arcsin(R[2, 0])
            rz = np.arctan2(R[1, 0] / np.cos(ry), R[0, 0] / np.cos(ry))
            rx = np.arctan2(R[2, 1] / np.cos(ry), R[2, 2] / np.cos(ry))
        else:
            # 处理奇异情况
            rz = 0
            if R[2, 0] < 0:
                ry = np.pi / 2
                rx = rz + np.arctan2(R[0, 1], R[0, 2])
            else:
                ry = -np.pi / 2
                rx = -rz + np.arctan2(-R[0, 1], -R[0, 2])

        return Pose6D(x, y, z, np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz))

    def calibrate(self) -> bool:
        """
        执行手眼标定计算

        Returns:
            是否标定成功
        """
        if len(self.samples) < 8:
            print("样本数量不足 (需要至少8组)，请增加样本")
            return False

        print(f"\n开始手眼标定计算，使用 {len(self.samples)} 组数据...")
        print("=" * 60)

        # 准备数据
        R_gripper2base = []  # 机器人末端到基座的旋转矩阵
        t_gripper2base = []  # 机器人末端到基座的平移向量
        R_target2cam = []  # 标定板到相机的旋转矩阵
        t_target2cam = []  # 标定板到相机的平移向量

        for sample in self.samples:
            # 机器人法兰盘在基坐标下的位姿 (baseTend)
            T_base2end = self.pose_to_homogeneous(sample.robot_pose)
            R_gripper2base.append(T_base2end[:3, :3])
            t_gripper2base.append(T_base2end[:3, 3])

            # 标定板在相机坐标系下的位姿 (cameraP)
            T_cam2board = self.pose_to_homogeneous(sample.board_pose)
            # 注意: OpenCV手眼标定使用 target2cam (标定板到相机)
            # 因此需要求逆
            T_board2cam = np.linalg.inv(T_cam2board)
            R_target2cam.append(T_board2cam[:3, :3])
            t_target2cam.append(T_board2cam[:3, 3])

        # 使用OpenCV进行手眼标定
        # 对于"眼在手外" (Eye-to-Hand)，使用 cv2.calibrateHandEye
        # 参数: (R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
        # 输出: 相机到基座的变换 (R_cam2base, t_cam2base)

        print("使用OpenCV进行手眼标定...")
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI  # 可以使用TSAI或PARK方法
        )

        # 构建齐次变换矩阵
        self.baseTcamera = np.eye(4)
        self.baseTcamera[:3, :3] = R_cam2base
        self.baseTcamera[:3, 3] = t_cam2base.flatten()

        # 验证结果
        self.validate_calibration()

        print("\n" + "=" * 60)
        print("标定计算完成！")
        print("相机到基座的变换矩阵 (baseTcamera):")
        print(self.baseTcamera)

        # 转换为位姿显示
        pose = self.homogeneous_to_pose(self.baseTcamera)
        print(f"\n转换为位姿: X={pose.x:.3f}, Y={pose.y:.3f}, Z={pose.z:.3f}, "
              f"Rx={pose.rx:.3f}, Ry={pose.ry:.3f}, Rz={pose.rz:.3f}")

        return True

    # ==================== 标定验证 ====================
    def validate_calibration(self):
        """验证标定结果"""
        if self.baseTcamera is None:
            return

        print("\n验证标定结果...")
        self.reprojection_errors = []

        for sample in self.samples:
            # 机器人位姿 (baseTend)
            T_base2end = self.pose_to_homogeneous(sample.robot_pose)

            # 标定板位姿 (cameraP)
            T_cam2board = self.pose_to_homogeneous(sample.board_pose)

            # 预测: baseTboard = baseTcamera * cameraTboard
            T_base2board_pred = self.baseTcamera @ T_cam2board

            # 实际: baseTboard = baseTend * endTboard
            # 注意: endTboard需要事先测量，这里简化处理
            # 如果endTboard未知，可以通过多组数据计算

            # 计算重投影误差 (简化版)
            error = np.linalg.norm(T_base2board_pred[:3, 3] - T_base2end[:3, 3])
            self.reprojection_errors.append(error)

        avg_error = np.mean(self.reprojection_errors)
        max_error = np.max(self.reprojection_errors)

        print(f"平均平移误差: {avg_error:.3f} mm")
        print(f"最大平移误差: {max_error:.3f} mm")

    # ==================== 结果保存 ====================
    def save_results(self):
        """保存标定结果"""
        if self.baseTcamera is None:
            print("没有标定结果可保存")
            return

        # 保存矩阵
        np.save(self.save_dir / "baseTcamera.npy", self.baseTcamera)

        # 保存为JSON (位姿格式)
        pose = self.homogeneous_to_pose(self.baseTcamera)
        result = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self.samples),
            "baseTcamera": {
                "matrix": self.baseTcamera.tolist(),
                "pose": {
                    "x": pose.x, "y": pose.y, "z": pose.z,
                    "rx": pose.rx, "ry": pose.ry, "rz": pose.rz
                }
            },
            "reprojection_errors": self.reprojection_errors,
            "avg_error": float(np.mean(self.reprojection_errors)) if self.reprojection_errors else 0
        }

        with open(self.save_dir / "calibration_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n标定结果已保存到: {self.save_dir}")

    # ==================== 加载标定结果 ====================
    def load_results(self) -> bool:
        """加载之前保存的标定结果"""
        matrix_path = self.save_dir / "baseTcamera.npy"
        if not matrix_path.exists():
            print("未找到标定结果文件")
            return False

        self.baseTcamera = np.load(matrix_path)
        print(f"已加载标定结果: {self.baseTcamera}")
        return True

    # ==================== 坐标转换 ====================
    def transform_camera_to_base(self, camera_pose: Pose6D) -> Pose6D:
        """
        将相机坐标系下的位姿转换到基坐标系

        Args:
            camera_pose: 相机坐标系下的位姿

        Returns:
            基坐标系下的位姿
        """
        if self.baseTcamera is None:
            raise ValueError("未加载标定结果")

        T_cam2obj = self.pose_to_homogeneous(camera_pose)
        T_base2obj = self.baseTcamera @ T_cam2obj
        return self.homogeneous_to_pose(T_base2obj)

    def transform_base_to_camera(self, base_pose: Pose6D) -> Pose6D:
        """
        将基坐标系下的位姿转换到相机坐标系

        Args:
            base_pose: 基坐标系下的位姿

        Returns:
            相机坐标系下的位姿
        """
        if self.baseTcamera is None:
            raise ValueError("未加载标定结果")

        T_base2obj = self.pose_to_homogeneous(base_pose)
        T_cam2base = np.linalg.inv(self.baseTcamera)
        T_cam2obj = T_cam2base @ T_base2obj
        return self.homogeneous_to_pose(T_cam2obj)


# ==================== 主程序 ====================
def main():
    """主程序"""
    print("=" * 60)
    print("华沿机器人 + OAK相机 手眼标定程序")
    print("构型: 眼在手外 (Eye-to-Hand)")
    print("=" * 60)

    # 创建标定器实例
    calibrator = HandEyeCalibration(robot_ip="192.168.0.10", port=10003)

    try:
        # 1. 连接机器人
        if not calibrator.connect_robot():
            print("机器人连接失败，请检查网络和控制器状态")
            return

        # 2. 采集标定数据
        print("\n请确保标定板已固定在机器人末端，并在OAK相机视野内")
        input("准备就绪后按Enter键继续...")

        if not calibrator.collect_samples(num_samples=15):
            print("标定数据采集失败")
            return

        # 3. 执行标定计算
        if not calibrator.calibrate():
            print("手眼标定计算失败")
            return

        # 4. 保存结果
        calibrator.save_results()

        # 5. 测试坐标转换
        print("\n" + "=" * 60)
        print("测试坐标转换...")

        # 模拟相机识别到目标
        test_camera_pose = Pose6D(x=100, y=50, z=200, rx=10, ry=5, rz=0)
        print(f"相机坐标下的目标: {test_camera_pose.to_list()}")

        # 转换到基坐标系
        base_pose = calibrator.transform_camera_to_base(test_camera_pose)
        print(f"基坐标下的目标: {base_pose.to_list()}")

        print("\n手眼标定完成！")

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 断开机器人连接
        calibrator.disconnect_robot()


if __name__ == "__main__":
    main()
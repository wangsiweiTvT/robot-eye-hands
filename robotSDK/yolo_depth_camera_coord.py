#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 目标检测 + OAK-D 深度 → 相机坐标系三维坐标

功能:
  1. 使用 YOLO (Ultralytics) 检测图像中的目标
  2. 从 OAK-D 双目相机获取深度图
  3. 计算每个检测目标的中心点在相机坐标系下的三维坐标 (Xc, Yc, Zc)
  4. 实时可视化显示

坐标系说明:
    相机坐标系 (OAK-D, 以左目CAM_B为对齐参考):
        Xc: 向右 (图像宽度方向)
        Yc: 向下 (图像高度方向)
        Zc: 向前 (深度方向，垂直于成像平面)

    针孔相机模型 (像素 → 相机坐标):
        Xc = (u - cx) * Z / fx
        Yc = (v - cy) * Z / fy
        Zc = Z

Author: Robot Vision Project
"""

import cv2
import numpy as np
import depthai as dai
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List
import time
import json
from datetime import datetime

# ============================================================
#  数据结构
# ============================================================
@dataclass
class DetectedObject:
    """检测到的目标及其三维信息"""
    # 边界框 (像素坐标)
    x1: int
    y1: int
    x2: int
    y2: int
    # 中心点 (像素坐标)
    center_x: float
    center_y: float
    # 类别 & 置信度
    class_name: str
    confidence: float
    # 深度信息
    depth_mm: float          # 深度值 (mm)
    depth_confidence: float  # 深度置信度 [0, 1]
    # 相机坐标系三维坐标 (mm)
    camera_coord: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 机械臂基座坐标系完整位姿 [X, Y, Z, RX, RY, RZ] (mm + 度)
    # 位置来自 camera_to_base 变换，姿态来自当前 TCP_111 朝向
    robot_pose: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ============================================================
#  深度平滑器 (可选，用于减少深度抖动)
# ============================================================
class DepthSmoother:
    """
    自适应深度平滑器
    对同一目标的深度序列进行加权中值滤波 + 离群值抑制
    """
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: List[float] = []

    def update(self, raw_depth: float) -> float:
        if raw_depth <= 0:
            return 0.0

        self.history.append(raw_depth)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 3:
            return raw_depth

        arr = np.array(self.history)
        # 剔除 >2倍标准差 的离群值
        mean, std = np.mean(arr), np.std(arr)
        filtered = arr[np.abs(arr - mean) <= 2 * std]
        if len(filtered) == 0:
            return float(np.median(arr))

        # 对剩余值加权平均 (越新权重越大)
        weights = np.linspace(0.5, 1.0, len(filtered))
        smoothed = np.average(filtered, weights=weights)
        return float(smoothed)


# ============================================================
#  OAK-D 深度相机模块
# ============================================================
class OAKDCamera:
    """
    OAK-D 双目深度相机封装

    提供:
      - 左目彩色/灰度图
      - 对齐到左目的深度图 (mm)
      - 像素→相机坐标转换
    """

    def __init__(self, fps: int = 30, ip: Optional[str] = None,
                 calib_path: str = "stereo_calibration_results.npz"):
        """
        Args:
            fps: 相机帧率
            ip: OAK相机IP地址 (None=USB连接, IP=网络连接)
            calib_path: 双目标定结果，读取 K_l 作为左目内参 (640x400)
        """
        self.fps = fps

        # ========== 工作分辨率 ==========
        self.left_width = 640
        self.left_height = 400

        # 深度图尺寸 (对齐到左目)
        self.depth_width = 640
        self.depth_height = 400

        # ========== 加载左目内参 (640x400, 无需缩放) ==========
        self.cx, self.cy, self.fx, self.fy = self._load_intrinsics(calib_path)

        # 构建 pipeline 并启动设备
        self.pipeline = self._build_pipeline(fps)

        if ip:
            self.device = dai.Device(self.pipeline, dai.DeviceInfo(ip))
        else:
            self.device = dai.Device(self.pipeline)

        # 获取输出队列
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        self.q_left = self.device.getOutputQueue("left", maxSize=4, blocking=False)

        print(f"✅ OAK-D 相机启动成功 ({self.left_width}x{self.left_height})")
        print(f"   内参: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

    @staticmethod
    def _load_intrinsics(calib_path: str) -> Tuple[float, float, float, float]:
        """从 stereo_calibration_results.npz 读取左目内参 K_l (640x400)"""
        try:
            data = np.load(calib_path)
            K = data['K_l']
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            print(f"  📷 加载标定文件: {calib_path}")
            print(f"     内参: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
            print(f"     分辨率: {data.get('image_size', 'N/A')}")
            return cx, cy, fx, fy
        except Exception as e:
            print(f"  ⚠️ 加载标定失败 ({e})，使用默认内参")
            return 341.26, 205.47, 398.44, 398.02

    def _build_pipeline(self, fps: int) -> dai.Pipeline:
        """构建 OAK-D pipeline: 左目(640x400) + 双目深度"""
        pipeline = dai.Pipeline()

        # --- 双目相机 (400P = 640x400) ---
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)   # 左目
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # 右目
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(fps)
        mono_right.setFps(fps)

        # --- 立体深度节点 ---
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        stereo.setOutputSize(self.depth_width, self.depth_height)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.initialConfig.setConfidenceThreshold(200)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # --- 输出流 ---
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("left")
        mono_left.out.link(xout_left.input)

        return pipeline

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取一帧左目图像 + 深度图

        Returns:
            left_frame:  左目灰度图 (H, W) 或 None
            depth_mm:    深度图 (H, W) 单位 mm，无效点为0
        """
        try:
            depth_data = self.q_depth.get()
            left_data = self.q_left.get()

            left_frame = left_data.getCvFrame()
            depth_mm = depth_data.getFrame().astype(np.float32)

            # 深度图滤波 + 限幅 (阈值与 dep_main.py 一致)
            depth_mm = cv2.medianBlur(depth_mm.astype(np.uint16), 5).astype(np.float32)
            depth_mm[(depth_mm < 100) | (depth_mm > 5000)] = 0

            return left_frame, depth_mm

        except Exception as e:
            print(f"  ⚠️ 获取帧失败: {e}")
            return None, None

    def get_object_depth(self, depth_map: np.ndarray,
                         bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        从深度图中提取目标区域的深度值 (dep_main.py 方法)

        策略:
          1. 取目标中心区域 (60%内缩) + 上下区域的有效深度
          2. 对所有有效深度排序，取前15%最小值求平均
             (因为深度误差通常是正值——反射/噪声导致深度偏大)
          3. 根据变异系数评估深度置信度

        Args:
            depth_map: 深度图 (H, W) mm
            bbox: (x1, y1, x2, y2) 目标边界框

        Returns:
            (depth_value, confidence):
                depth_value: 估计的深度值 (mm)，无效返回0
                confidence:  深度置信度 [0, 1]
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape

        x1 = max(0, min(x1, w - 2))
        y1 = max(0, min(y1, h - 2))
        x2 = max(x1 + 2, min(x2, w))
        y2 = max(y1 + 2, min(y2, h))

        box_h = y2 - y1
        box_w = x2 - x1

        regions = []

        # 中心区域
        margin_x = int(box_w * 0.2)
        margin_y = int(box_h * 0.2)
        cx1 = x1 + margin_x
        cy1 = y1 + margin_y
        cx2 = x2 - margin_x
        cy2 = y2 - margin_y
        if cx2 > cx1 and cy2 > cy1:
            roi = depth_map[cy1:cy2, cx1:cx2]
            valid = roi[(roi > 100) & (roi < 5000)]
            if len(valid) > 0:
                regions.append(valid)

        # 上半部分
        top_roi = depth_map[y1:y1 + box_h // 3, x1:x2]
        valid_top = top_roi[(top_roi > 100) & (top_roi < 5000)]
        if len(valid_top) > 5:
            regions.append(valid_top)

        # 下半部分
        bottom_roi = depth_map[y1 + 2 * box_h // 3:y2, x1:x2]
        valid_bottom = bottom_roi[(bottom_roi > 100) & (bottom_roi < 5000)]
        if len(valid_bottom) > 5:
            regions.append(valid_bottom)

        if len(regions) == 0:
            return 0.0, 0.0

        all_valid = np.concatenate(regions)
        if len(all_valid) < 5:
            return 0.0, 0.0

        # 取前15%最小值 (抑制噪声导致的偏大深度)
        sorted_depth = np.sort(all_valid)
        min_count = max(5, int(len(sorted_depth) * 0.15))
        depth_value = float(np.mean(sorted_depth[:min_count]))

        # 置信度: 基于变异系数 CV = std/mean
        if len(sorted_depth) > 10:
            valid_for_conf = sorted_depth[:min_count]
            std = np.std(valid_for_conf)
            mean = np.mean(valid_for_conf)
            cv_ratio = std / (mean + 1e-6)
            confidence = max(0.0, min(1.0, 1.0 - cv_ratio * 2))
        else:
            confidence = 0.5

        return depth_value, confidence

    def pixel_to_camera_coord(self, pixel_x: float, pixel_y: float,
                               depth_mm: float) -> Tuple[float, float, float]:
        """
        像素坐标 (u, v) + 深度 Z → 相机坐标系三维坐标 (Xc, Yc, Zc)

        针孔相机模型:
            Xc = (u - cx) * Z / fx
            Yc = (v - cy) * Z / fy
            Zc = Z

        原理:
            像素坐标 (u, v) 是物体在成像平面上的投影，
            通过内参 (fx, fy, cx, cy) 和深度 Z 可以反投影到三维空间。

        Args:
            pixel_x:   像素坐标 u (宽度方向，单位 pixel)
            pixel_y:   像素坐标 v (高度方向，单位 pixel)
            depth_mm:  深度值 (单位 mm)

        Returns:
            (Xc, Yc, Zc): 相机坐标系下的三维坐标 (单位 mm)
                           depth_mm <= 0 时返回 (0, 0, 0)
        """
        if depth_mm <= 0:
            return (0.0, 0.0, 0.0)

        # 反投影: 像素 → 归一化平面 → 乘以深度
        Xc = (pixel_x - self.cx) * depth_mm / self.fx
        Yc = (pixel_y - self.cy) * depth_mm / self.fy
        Zc = depth_mm

        return (Xc, Yc, Zc)

    def close(self):
        """释放相机资源"""
        if hasattr(self, 'device'):
            self.device.close()
            print("  ✓ OAK-D 相机已关闭")


# ============================================================
#  相机 → 机械臂基座坐标变换 (手眼标定 + TCP_111)
# ============================================================
class RobotCoordTransformer:
    """
    利用手眼标定结果，将相机坐标系下的三维点变换到机械臂基座坐标系。

    变换链:
        相机坐标 (Xc, Yc, Zc)
            ↓  endTcam           (手眼标定: 相机→TCP_111)
        TCP_111 坐标
            ↓  baseTtcp111       (HRIF_ReadActTcpPos: TCP_111→基座)
        机械臂基座坐标 (Xb, Yb, Zb)

    核心公式:
        P_base = baseTtcp111 @ endTcam @ P_cam

    标定文件:
        ./calibration_result_hand_in_eye/endTcam.npy
        endTcam = 相机坐标系 → TCP_111 的变换矩阵

    Mock 模式:
        当无法连接机器人时，使用预设的模拟 TCP 位姿进行测试。
    """

    def __init__(self, calib_dir: str = "./calibration_result_hand_in_eye",
                 use_mock: bool = False):
        self.calib_dir = Path(calib_dir)
        self.endTcam = None          # 手眼标定: 相机→TCP_111 (4x4)
        self.cps = None
        self.connected = False
        self.calibrated = False

        # Mock 模式
        self.use_mock = use_mock
        self.MOCK_TCP_POSE = [0, 0, 0, 0, 0, 0]
        self.mock_pose_4x4 = np.eye(4)

        # 机器人参数
        self.robot_ip = "192.168.0.10"
        self.robot_port = 10003
        self.box_id = 0
        self.rbt_id = 0

        self._load_calibration()

    # ---------- 初始化 ----------

    def _load_calibration(self) -> bool:
        """加载手眼标定结果 endTcam.npy"""
        try:
            calib_path = self.calib_dir / "endTcam.npy"
            if not calib_path.exists():
                print(f"  ⚠️ 未找到标定文件: {calib_path}")
                return False
            self.endTcam = np.load(calib_path)
            self.calibrated = True
            trans = self.endTcam[:3, 3]
            print(f"  ✅ 加载手眼标定: {calib_path}")
            print(f"     平移: X={trans[0]:.1f}  Y={trans[1]:.1f}  Z={trans[2]:.1f} mm")
            return True
        except Exception as e:
            print(f"  ❌ 加载手眼标定失败: {e}")
            return False

    def connect_robot(self) -> bool:
        """连接华沿机器人"""
        try:
            from CPS import CPSClient
            self.cps = CPSClient()
            print(f"  连接机器人 {self.robot_ip}:{self.robot_port} ...")
            ret = self.cps.HRIF_Connect(self.box_id, self.robot_ip, self.robot_port)
            if ret != 0:
                print(f"  ❌ 连接失败: {ret}")
                return False
            for i in range(30):
                result = []
                ret = self.cps.HRIF_IsControllerStarted(self.box_id, result)
                if ret == 0 and len(result) > 0 and int(result[0]) == 1:
                    break
                time.sleep(0.5)
            self.cps.HRIF_GrpEnable(self.box_id, self.rbt_id)
            self.connected = True
            print("  ✅ 机器人连接成功")
            return True
        except ImportError:
            print("  ⚠️ CPS SDK 未安装，仅输出相机坐标")
            return False
        except Exception as e:
            print(f"  ❌ 机器人连接失败: {e}")
            return False

    # ---------- 读取机器人位姿 ----------

    def read_tcp_pose(self) -> Optional[np.ndarray]:
        """读取当前 TCP_111 在基座下的位姿 (4x4)"""
        if self.use_mock:
            return self.mock_pose_4x4.copy()
        if not self.connected or self.cps is None:
            return None
        try:
            result = []
            ret = self.cps.HRIF_ReadActTcpPos(self.box_id, self.rbt_id, result)
            if ret != 0 or len(result) < 6:
                return None
            return self._euler_to_matrix(
                float(result[0]), float(result[1]), float(result[2]),
                float(result[3]), float(result[4]), float(result[5])
            )
        except Exception as e:
            print(f"  ⚠️ 读取 TCP 位姿失败: {e}")
            return None

    def read_tcp_euler(self) -> Optional[Tuple[float, float, float]]:
        """读取当前 TCP_111 朝向角 (RX, RY, RZ)"""
        if self.use_mock:
            return (self.MOCK_TCP_POSE[3], self.MOCK_TCP_POSE[4], self.MOCK_TCP_POSE[5])
        if not self.connected or self.cps is None:
            return None
        try:
            result = []
            ret = self.cps.HRIF_ReadActTcpPos(self.box_id, self.rbt_id, result)
            if ret != 0 or len(result) < 6:
                return None
            return (float(result[3]), float(result[4]), float(result[5]))
        except Exception as e:
            print(f"  ⚠️ 读取 TCP 朝向失败: {e}")
            return None

    # ---------- 坐标变换核心 ----------

    def camera_to_base(self, camera_coord: Tuple[float, float, float],
                        base_T_tcp111: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        相机坐标 → 机械臂基座坐标

        P_base = baseTtcp111 @ endTcam @ P_cam

        Args:
            camera_coord: (Xc, Yc, Zc) 相机坐标 (mm)
            base_T_tcp111: TCP_111 在基座下的 4x4 矩阵，None 时自动读取

        Returns:
            (Xb, Yb, Zb) 基座坐标 (mm)
        """
        if not self.calibrated or self.endTcam is None:
            return (0.0, 0.0, 0.0)
        if base_T_tcp111 is None:
            base_T_tcp111 = self.read_tcp_pose()
            if base_T_tcp111 is None:
                return (0.0, 0.0, 0.0)
        P_cam = np.array([camera_coord[0], camera_coord[1], camera_coord[2], 1.0])
        P_base = base_T_tcp111 @ (self.endTcam @ P_cam)
        return (float(P_base[0]), float(P_base[1]), float(P_base[2]))

    # ---------- 工具方法 ----------

    @staticmethod
    def _euler_to_matrix(x: float, y: float, z: float,
                         rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
        """欧拉角 (度, ZYX顺序) + 平移 → 4x4"""
        rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx),  np.cos(rx)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0, 0, 1]])

        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def _matrix_to_euler(T: np.ndarray) -> dict:
        """4x4 → {x, y, z, rx, ry, rz} (ZYX)"""
        x, y, z = T[:3, 3]
        R = T[:3, :3]
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0
        return {"x": float(x), "y": float(y), "z": float(z),
                "rx": np.rad2deg(rx), "ry": np.rad2deg(ry), "rz": np.rad2deg(rz)}

    def disconnect(self):
        if self.connected and self.cps is not None:
            try:
                self.cps.HRIF_DisConnect(self.box_id)
                print("  ✓ 机器人已断开")
            except Exception:
                pass
            self.connected = False


# ============================================================
#  坐标保存 (JSON)
# ============================================================
def save_detections_to_json(detected_objects: List[DetectedObject],
                             output_dir: str = "detections",
                             filename: Optional[str] = None) -> Optional[str]:
    """
    将检测结果 (含相机坐标 + 基座坐标) 保存为 JSON 文件

    每行一个目标，方便模块化测试和后续处理。

    JSON 格式:
    {
        "timestamp": "2026-06-20T15:30:00",
        "num_objects": 2,
        "objects": [
            {
                "class_name": "bottle",
                "confidence": 0.95,
                "pixel_center": [320.5, 240.0],
                "depth_mm": 850.3,
                "depth_confidence": 0.92,
                "camera_coord_mm": [45.2, -12.8, 850.3],
                "robot_pose": [312.5, -45.6, 125.8, -10.2, -74.0, 10.4]
                # robot_pose: [X, Y, Z, RX, RY, RZ] → 可直接传入 move_to_cartesian()
            },
            ...
        ]
    }

    Args:
        detected_objects: 检测到的目标列表
        output_dir: 输出目录 (自动创建)
        filename: 自定义文件名，None 则自动生成

    Returns:
        保存的文件路径，失败返回 None
    """
    if not detected_objects:
        print("  ⚠️ 没有检测到目标，跳过保存")
        return None

    # 确保输出目录存在
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    # 生成文件名
    if filename is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp_str}.json"

    filepath = out_dir / filename

    # 构建 JSON 数据
    objects_data = []
    for obj in detected_objects:
        obj_dict = {
            "class_name": obj.class_name,
            "confidence": round(obj.confidence, 4),
            "pixel_center": [round(obj.center_x, 1), round(obj.center_y, 1)],
            "bbox": [obj.x1, obj.y1, obj.x2, obj.y2],
            "depth_mm": round(obj.depth_mm, 1),
            "depth_confidence": round(obj.depth_confidence, 4),
            "camera_coord_mm": [
                round(obj.camera_coord[0], 1),
                round(obj.camera_coord[1], 1),
                round(obj.camera_coord[2], 1)
            ],
            # [X, Y, Z, RX, RY, RZ] — 可直接用于 move_to_cartesian()
            "robot_pose": [
                round(obj.robot_pose[0], 1),
                round(obj.robot_pose[1], 1),
                round(obj.robot_pose[2], 1),
                round(obj.robot_pose[3], 1),
                round(obj.robot_pose[4], 1),
                round(obj.robot_pose[5], 1)
            ],
        }
        objects_data.append(obj_dict)

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_objects": len(objects_data),
        "objects": objects_data,
    }

    # 写入 JSON (格式化输出)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  💾 坐标已保存: {filepath}")
        print(f"     目标数: {len(objects_data)}")
        return str(filepath)
    except Exception as e:
        print(f"  ❌ 保存失败: {e}")
        return None


def main():
    print("=" * 65)
    print("  YOLO 目标检测 → 深度 → 相机坐标 → 机械臂基座坐标")
    print("=" * 65)

    # ========== 配置参数 ==========
    YOLO_MODEL_PATH = "best.pt"          # YOLO 模型权重
    CONF_THRESHOLD = 0.3                 # 检测置信度阈值
    CAMERA_FPS = 30                      # 相机帧率
    CAMERA_IP = "169.254.1.222"          # OAK-D IP (网络连接)
    ROBOT_IP = "192.168.0.10"            # 华沿机器人 IP
    USE_ROBOT = False                    # 是否连接真实机器人
    USE_MOCK = True                      # 是否使用模拟位姿 (无机器人时)

    # ========== 手动设置 TCP_111 位姿 (Mock 模式用) ==========
    # 每次运行前，把机器人的实际 TCP_111 位姿填在这里
    # 格式: [X, Y, Z, RX, RY, RZ]
    MOCK_TCP_POSE = [-384.743, 36.632, 628.872, -174.439, -10.757, 176.110]

    # ========== 初始化 YOLO ==========
    print("\n[1] 加载 YOLO 模型...")
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_PATH)
        print(f"    ✅ YOLO 加载成功: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"    ❌ YOLO 加载失败: {e}")
        return

    # ========== 初始化 OAK-D 相机 ==========
    print("\n[2] 初始化 OAK-D 相机...")
    try:
        camera = OAKDCamera(fps=CAMERA_FPS, ip=CAMERA_IP)
    except Exception as e:
        print(f"    ❌ 相机初始化失败: {e}")
        return

    # ========== 初始化坐标变换 (手眼标定 + TCP_111) ==========
    print("\n[3] 初始化坐标变换...")

    if USE_MOCK:
        transformer = RobotCoordTransformer(use_mock=True)
        # 使用配置区的位姿 (可在运行前修改)
        transformer.MOCK_TCP_POSE = MOCK_TCP_POSE
        transformer.mock_pose_4x4 = transformer._euler_to_matrix(*MOCK_TCP_POSE)
        print("    🧪 Mock 模式: 使用配置的 TCP 位姿")
        print(f"       TCP_111: X={MOCK_TCP_POSE[0]:.1f} Y={MOCK_TCP_POSE[1]:.1f} Z={MOCK_TCP_POSE[2]:.1f}")
        print(f"                RX={MOCK_TCP_POSE[3]:.1f} RY={MOCK_TCP_POSE[4]:.1f} RZ={MOCK_TCP_POSE[5]:.1f}")
    else:
        transformer = RobotCoordTransformer()

    if not transformer.calibrated:
        print("    ⚠️ 手眼标定未加载")
        print("    ❌ 无法继续: 需要手眼标定结果 endTcam.npy")
        return

    if USE_ROBOT and not USE_MOCK:
        transformer.robot_ip = ROBOT_IP
        transformer.connect_robot()

    # ========== 主循环 ==========
    print("\n[4] 启动实时检测...")
    print("    Controls:")
    print("      'q'   - 退出")
    print("      'd'   - 调试信息 (打印所有坐标)")
    print("      'w'   - 保存坐标到 JSON 文件")
    print("      's'   - 保存当前帧截图")
    print("      'p'   - 手动输入 TCP_111 位姿")
    print("      'r'   - 刷新机器人位姿")
    print("=" * 65)

    # 用于深度平滑 (每个目标独立平滑)
    smoother_map = {}

    # 缓存当前机器人位姿 (避免每帧读取)
    current_base_T_tcp111 = None
    pose_timestamp = 0

    try:
        while True:
            # ---- 获取帧 ----
            left_frame, depth_mm = camera.get_frames()
            if left_frame is None:
                continue

            # 灰度 → BGR (用于显示和YOLO)
            frame_rgb = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
            display = frame_rgb.copy()

            # ---- YOLO 检测 ----
            results = model(frame_rgb, conf=CONF_THRESHOLD, verbose=False)

            detected_objects: List[DetectedObject] = []
            detected_tcp111: List[Tuple[float, float, float]] = []

            for r in results:
                if r.boxes is None:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    # ---- 获取目标深度 ----
                    depth_val, depth_conf = camera.get_object_depth(
                        depth_mm, (x1, y1, x2, y2)
                    )

                    # ---- 深度平滑 ----
                    if depth_val > 0:
                        smoother_key = len(detected_objects)
                        if smoother_key not in smoother_map:
                            smoother_map[smoother_key] = DepthSmoother()
                        depth_val = smoother_map[smoother_key].update(depth_val)

                    # ---- 计算相机坐标系三维坐标 ----
                    cam_coord = camera.pixel_to_camera_coord(
                        center_x, center_y, depth_val
                    )

                    # ---- 计算机器人基座坐标系完整位姿 [X, Y, Z, RX, RY, RZ] ----
                    robot_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    tcp111_coord = (0.0, 0.0, 0.0)  # 中间 TCP_111 坐标
                    if transformer.calibrated and depth_val > 0:
                        # ---- 相机 → TCP_111 (验证 endTcam) ----
                        P_cam_h = np.array([cam_coord[0], cam_coord[1], cam_coord[2], 1.0])
                        P_tcp111 = transformer.endTcam @ P_cam_h
                        tcp111_coord = (P_tcp111[0], P_tcp111[1], P_tcp111[2])

                        # 位置: TCP_111 → 基座
                        pos = transformer.camera_to_base(
                            cam_coord, current_base_T_tcp111
                        )
                        # 姿态: 使用当前 TCP_111 的朝向
                        ori = transformer.read_tcp_euler()
                        if ori is not None and pos[2] > 0:
                            robot_pose = (pos[0], pos[1], pos[2], ori[0], ori[1], ori[2])

                    obj = DetectedObject(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        center_x=center_x, center_y=center_y,
                        class_name=model.names[int(clses[i])],
                        confidence=float(confs[i]),
                        depth_mm=depth_val,
                        depth_confidence=depth_conf,
                        camera_coord=cam_coord,
                        robot_pose=robot_pose,
                    )
                    detected_objects.append(obj)
                    detected_tcp111.append(tcp111_coord)

            # ========== 终端输出 ==========
            if detected_objects:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 检测到 {len(detected_objects)} 个目标:")
                for idx, (obj, tc) in enumerate(zip(detected_objects, detected_tcp111)):
                    cx, cy, cz = obj.camera_coord
                    px, py, pz, prx, pry, prz = obj.robot_pose
                    cam = f"Cam({cx:.0f},{cy:.0f},{cz:.0f})"
                    tcp = f"TCP({tc[0]:.0f},{tc[1]:.0f},{tc[2]:.0f})"
                    if pz > 0:
                        base = f"Base({px:.1f},{py:.1f},{pz:.1f},{prx:.1f},{pry:.1f},{prz:.1f})"
                    else:
                        base = "Base(--- no pose ---)"
                    print(f"  [{idx+1}] {obj.class_name} d={obj.depth_mm:.0f}mm | {cam} → {tcp} → {base}")

            # ========== 可视化 ==========

            # ---- 绘制深度图 ----
            if depth_mm is not None:
                depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            else:
                depth_colored = np.zeros((400, 640, 3), dtype=np.uint8)

            # ---- 绘制检测结果 ----
            for obj in detected_objects:
                # 边界框
                cv2.rectangle(display, (obj.x1, obj.y1), (obj.x2, obj.y2),
                              (0, 255, 0), 2)

                # 中心点
                cv2.circle(display, (int(obj.center_x), int(obj.center_y)),
                           5, (0, 0, 255), -1)

                # 标签: 类别 + 置信度
                label = f"{obj.class_name} {obj.confidence:.2f}"
                cv2.putText(display, label, (obj.x1, obj.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 深度信息
                if obj.depth_mm > 0:
                    # 深度值 + 置信度
                    depth_text = f"Depth: {obj.depth_mm:.0f} mm (conf:{obj.depth_confidence:.2f})"
                    cv2.putText(display, depth_text, (obj.x1, obj.y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                    # 相机坐标
                    cam_str = f"Cam:({obj.camera_coord[0]:.0f},{obj.camera_coord[1]:.0f},{obj.camera_coord[2]:.0f})"
                    cv2.putText(display, cam_str, (obj.x1, obj.y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    # 机器人基座完整位姿 [X, Y, Z, RX, RY, RZ]
                    px, py, pz, prx, pry, prz = obj.robot_pose
                    if pz > 0:
                        rob_str = f"Base:({px:.0f},{py:.0f},{pz:.0f})"
                        cv2.putText(display, rob_str, (obj.x1, obj.y2 + 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # ---- HUD 信息 ----
            cv2.putText(display, f"Detected: {len(detected_objects)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if transformer.use_mock:
                status = "Mock"
            elif transformer.connected:
                status = "Robot:ON"
            else:
                status = "Robot:OFF"
            cv2.putText(display, f"q:quit w:save p:pose s:screenshot | {status}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # ---- 显示 ----
            depth_small = cv2.resize(depth_colored,
                                      (display.shape[1], display.shape[0] // 3))
            combined = np.vstack([display, depth_small])

            cv2.namedWindow("YOLO + Depth → Robot Base Coords", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLO + Depth → Robot Base Coords", combined)

            # ========== 键盘响应 ==========
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n    👋 用户退出")
                break

            elif key == ord('d'):
                print(f"\n{'=' * 70}")
                print(f"  📊 调试信息 (检测到 {len(detected_objects)} 个目标)")
                print(f"{'=' * 70}")
                for idx, (obj, tc) in enumerate(zip(detected_objects, detected_tcp111)):
                    cx, cy, cz = obj.camera_coord
                    px, py, pz, prx, pry, prz = obj.robot_pose
                    print(f"  [{idx + 1}] {obj.class_name}")
                    print(f"      像素中心: ({obj.center_x:.1f}, {obj.center_y:.1f})")
                    print(f"      深度: {obj.depth_mm:.1f} mm (置信度: {obj.depth_confidence:.2f})")
                    print(f"      📍 相机坐标: Xc={cx:.1f}  Yc={cy:.1f}  Zc={cz:.1f}")
                    print(f"      🔗 TCP_111 坐标: Xt={tc[0]:.1f}  Yt={tc[1]:.1f}  Zt={tc[2]:.1f}")
                    if pz > 0:
                        print(f"      🦾 基座位姿: X={px:.1f} Y={py:.1f} Z={pz:.1f}")
                        print(f"                  RX={prx:.1f} RY={pry:.1f} RZ={prz:.1f}")
                        print(f"      → 可直接用于 move_to_cartesian({[round(px,1), round(py,1), round(pz,1), round(prx,1), round(pry,1), round(prz,1)]})")
                    print()

            elif key == ord('p'):
                print("\n" + "=" * 60)
                print("  📝 手动输入 TCP_111 位姿 (Mock 模式)")
                print("=" * 60)
                print("  输入格式: X Y Z RX RY RZ")
                print("  示例: -696.2 -25.5 895.0 -10.2 -74.0 10.4")
                print("  输入 'q' 取消")
                try:
                    raw = input("  >> ").strip()
                    if raw.lower() == 'q':
                        print("  ⏹ 已取消")
                    else:
                        values = [float(v) for v in raw.split()]
                        if len(values) == 6:
                            # 更新 transformer 的 mock 位姿
                            transformer.MOCK_TCP_POSE = values
                            transformer.mock_pose_4x4 = transformer._euler_to_matrix(*values)
                            # Mock 模式下 mock_pose_4x4 就是 TCP_111 位姿
                            current_base_T_tcp111 = transformer.mock_pose_4x4
                            print(f"  ✅ TCP 位姿已更新:")
                            print(f"     X={values[0]:.1f} Y={values[1]:.1f} Z={values[2]:.1f}")
                            print(f"     RX={values[3]:.1f} RY={values[4]:.1f} RZ={values[5]:.1f}")
                        else:
                            print(f"  ❌ 需要6个值，收到 {len(values)} 个")
                except ValueError as e:
                    print(f"  ❌ 输入格式错误: {e}")
                print("=" * 60)

            elif key == ord('r'):
                print("  🔄 刷新 TCP 位姿...")
                current_base_T_tcp111 = transformer.read_tcp_pose()
                if current_base_T_tcp111 is not None:
                    pose = transformer._matrix_to_euler(current_base_T_tcp111)
                    print(f"     TCP_111: X={pose['x']:.1f} Y={pose['y']:.1f} Z={pose['z']:.1f}")
                    print(f"              RX={pose['rx']:.1f} RY={pose['ry']:.1f} RZ={pose['rz']:.1f}")
                else:
                    print("     ⚠️ 读取位姿失败")

            elif key == ord('w'):
                save_detections_to_json(detected_objects)

            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                filename = f"capture_{timestamp}.png"
                cv2.imwrite(filename, combined)
                print(f"    💾 截图已保存: {filename}")

    except KeyboardInterrupt:
        print("\n    ⚠️ 用户中断")
    except Exception as e:
        print(f"\n    ❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ---- 清理 ----
        cv2.destroyAllWindows()
        camera.close()
        transformer.disconnect()
        print("\n✅ 程序已退出")


if __name__ == "__main__":
    main()

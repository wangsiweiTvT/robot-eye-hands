#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定点运动测试 (笛卡尔坐标 + MoveJ)
目标: (-696.197, -25.541, 895.042, -10.245, -74.037, 10.355)
"""

import sys
import time
from typing import Tuple, Optional, List

# ==================== 导入华沿SDK ====================
try:
    from CPS import CPSClient
    print("✅ 华沿SDK 导入成功!")
except ImportError as e:
    print(f"❌ 华沿SDK 导入失败: {e}")
    sys.exit(1)


# ==================== 配置 ====================
ROBOT_IP = "192.168.0.10"
PORT = 10003
BOX_ID = 0
RBT_ID = 0

# ==================== 目标位置 ====================
TARGET_POSE = [-696.197, -25.541, 895.042, -10.245, -74.037, 10.355]


# ==================== 机器人控制器 ====================
class RobotController:
    def __init__(self):
        self.cps = CPSClient()
        self.box_id = BOX_ID
        self.rbt_id = RBT_ID
        self.is_connected = False

    def connect(self):
        print("  连接控制器...")
        ret = self.cps.HRIF_Connect(self.box_id, ROBOT_IP, PORT)
        if ret != 0:
            return False, f"连接失败, 错误码: {ret}"
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
            err_msg = self.get_error_str(ret)
            return False, f"使能失败, 错误码: {ret}, 描述: {err_msg}"
        print("  ✓ 已使能")

        self.is_connected = True
        return True, "连接成功"

    def disconnect(self):
        try:
            self.cps.HRIF_DisConnect(self.box_id)
            print("  ✓ 已断开连接")
        except:
            pass
        self.is_connected = False

    def get_error_str(self, error_code: int) -> str:
        try:
            result = []
            ret = self.cps.HRIF_GetErrorCodeStr(self.box_id, error_code, result)
            if ret == 0 and len(result) > 0:
                return result[0]
        except:
            pass
        return f"未知错误: {error_code}"

    def read_current_tcp(self):
        """读取当前TCP笛卡尔坐标"""
        result = []
        ret = self.cps.HRIF_ReadActTcpPos(self.box_id, self.rbt_id, result)
        if ret != 0:
            return False, None, f"读取失败, 错误码: {ret}"
        if len(result) < 6:
            return False, None, f"数据不足: {len(result)}"
        return True, {
            "x": float(result[0]), "y": float(result[1]), "z": float(result[2]),
            "rx": float(result[3]), "ry": float(result[4]), "rz": float(result[5])
        }, "成功"

    def read_current_joint(self):
        """读取当前关节位置"""
        result = []
        ret = self.cps.HRIF_ReadActPos(self.box_id, self.rbt_id, result)
        if ret != 0:
            return False, None, f"读取失败, 错误码: {ret}"
        if len(result) < 6:
            return False, None, f"数据不足: {len(result)}"
        return True, [float(x) for x in result[:6]], "成功"

    def move_to_cartesian(self, target_pose, velocity=20, acc=40, radius=0):
        """
        使用 MoveJ 运动到笛卡尔目标
        nIsUseJoint = 0
        """
        # 读取当前关节位置作为逆解参考
        ok, ref_joint, _ = self.read_current_joint()
        if not ok:
            return False, "无法读取当前关节位置作为参考"

        ret = self.cps.HRIF_MoveJ(
            self.box_id, self.rbt_id,
            target_pose,          # 笛卡尔目标
            ref_joint,            # 逆解参考关节
            "TCP_111",                # 工具坐标名称
            "Base",               # 用户坐标名称
            velocity,             # 速度 [°/s]
            acc,                  # 加速度 [°/s²]
            radius,               # 过渡半径 [mm]
            0,                    # nIsUseJoint=0: 使用笛卡尔坐标
            0, 0, 0,              # DI停止检测 (关闭)
            "0"                   # 命令ID
        )
        if ret != 0:
            err_msg = self.get_error_str(ret)
            return False, f"MoveJ失败, 错误码: {ret}, 描述: {err_msg}"
        return True, "MoveJ指令已发送"

    def wait_done(self, timeout=30):
        """等待运动完成"""
        start = time.time()
        result = []
        while time.time() - start < timeout:
            ret = self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
            if ret == 0 and len(result) > 0:
                if str(result[0]).lower() == "true":
                    return True, "运动完成"
            time.sleep(0.05)
        return False, "运动超时"


# ==================== 测试函数 ====================

def print_pose(name: str, pose: dict):
    print(f"  {name}:")
    print(f"    X  = {pose['x']:8.3f} mm")
    print(f"    Y  = {pose['y']:8.3f} mm")
    print(f"    Z  = {pose['z']:8.3f} mm")
    print(f"    Rx = {pose['rx']:8.3f}°")
    print(f"    Ry = {pose['ry']:8.3f}°")
    print(f"    Rz = {pose['rz']:8.3f}°")


def print_pose_list(name: str, pose: List[float]):
    print(f"  {name}:")
    print(f"    X  = {pose[0]:8.3f} mm")
    print(f"    Y  = {pose[1]:8.3f} mm")
    print(f"    Z  = {pose[2]:8.3f} mm")
    print(f"    Rx = {pose[3]:8.3f}°")
    print(f"    Ry = {pose[4]:8.3f}°")
    print(f"    Rz = {pose[5]:8.3f}°")


def print_joint(name: str, joint: List[float]):
    print(f"  {name}:")
    print(f"    J1 = {joint[0]:8.3f}°")
    print(f"    J2 = {joint[1]:8.3f}°")
    print(f"    J3 = {joint[2]:8.3f}°")
    print(f"    J4 = {joint[3]:8.3f}°")
    print(f"    J5 = {joint[4]:8.3f}°")
    print(f"    J6 = {joint[5]:8.3f}°")


def test_connect():
    """测试1: 连接机器人"""
    print("\n" + "=" * 60)
    print("【测试1】连接机器人")
    print("=" * 60)

    robot = RobotController()
    success, msg = robot.connect()
    if success:
        print(f"✅ {msg}")
    else:
        print(f"❌ {msg}")
    robot.disconnect()
    return success


def test_read_current():
    """测试2: 读取当前位置"""
    print("\n" + "=" * 60)
    print("【测试2】读取当前位置")
    print("=" * 60)

    robot = RobotController()
    success, msg = robot.connect()
    if not success:
        print(f"❌ {msg}")
        return False

    # 读取笛卡尔位置
    ok, pose, msg = robot.read_current_tcp()
    if ok:
        print("  ✅ 当前笛卡尔位置:")
        print_pose("    ", pose)
    else:
        print(f"  ❌ {msg}")

    # 读取关节位置
    ok, joint, msg = robot.read_current_joint()
    if ok:
        print("\n  ✅ 当前关节位置:")
        print_joint("    ", joint)
    else:
        print(f"  ❌ {msg}")

    robot.disconnect()
    return True


def test_move_to_target():
    """测试3: 运动到目标点"""
    print("\n" + "=" * 60)
    print("【测试3】运动到目标点")
    print("=" * 60)

    print("  目标位置 (笛卡尔):")
    print_pose_list("    ", TARGET_POSE)

    robot = RobotController()
    success, msg = robot.connect()
    if not success:
        print(f"❌ {msg}")
        return False

    # 读取起始位置
    ok, start_pose, msg = robot.read_current_tcp()
    if ok:
        print("\n  起始位置:")
        print_pose("    ", start_pose)
    else:
        print(f"\n  ❌ 读取起始位置失败: {msg}")
        robot.disconnect()
        return False

    # 读取起始关节
    ok, start_joint, msg = robot.read_current_joint()
    if ok:
        print("\n  起始关节:")
        print_joint("    ", start_joint)

    # 执行运动
    print("\n  执行 MoveJ (nIsUseJoint=0)...")
    print(f"    速度: 20°/s, 加速度: 40°/s²")
    success, msg = robot.move_to_cartesian(TARGET_POSE, velocity=20, acc=40)
    if not success:
        print(f"  ❌ {msg}")
        robot.disconnect()
        return False
    print(f"  ✅ {msg}")

    # 等待运动完成
    print("\n  等待运动完成...")
    done, msg = robot.wait_done(timeout=60)
    if not done:
        print(f"  ❌ {msg}")
        robot.disconnect()
        return False
    print(f"  ✅ {msg}")

    # 读取到达位置
    ok, end_pose, msg = robot.read_current_tcp()
    if not ok:
        print(f"  ❌ 读取到达位置失败: {msg}")
        robot.disconnect()
        return False

    print("\n  到达位置:")
    print_pose("    ", end_pose)

    # 读取到达关节
    ok, end_joint, msg = robot.read_current_joint()
    if ok:
        print("\n  到达关节:")
        print_joint("    ", end_joint)

    # 计算误差
    dx = end_pose['x'] - TARGET_POSE[0]
    dy = end_pose['y'] - TARGET_POSE[1]
    dz = end_pose['z'] - TARGET_POSE[2]
    drx = end_pose['rx'] - TARGET_POSE[3]
    dry = end_pose['ry'] - TARGET_POSE[4]
    drz = end_pose['rz'] - TARGET_POSE[5]

    error_pos = (dx**2 + dy**2 + dz**2) ** 0.5
    error_ori = (drx**2 + dry**2 + drz**2) ** 0.5

    print("\n  📊 误差分析:")
    print(f"    位置误差: X={dx:+.3f}mm Y={dy:+.3f}mm Z={dz:+.3f}mm")
    print(f"    姿态误差: Rx={drx:+.3f}° Ry={dry:+.3f}° Rz={drz:+.3f}°")
    print(f"    位置总误差: {error_pos:.3f}mm")
    print(f"    姿态总误差: {error_ori:.3f}°")

    # 判断
    if error_pos < 2.0 and error_ori < 1.0:
        print("\n  ✅ 运动成功到达目标点，精度良好")
    elif error_pos < 5.0:
        print("\n  ⚠ 位置有偏差，但仍在可接受范围内")
    else:
        print("\n  ❌ 位置偏差较大，请检查TCP或运动参数")

    robot.disconnect()
    return True


def test_move_with_different_speed():
    """测试4: 不同速度参数测试"""
    print("\n" + "=" * 60)
    print("【测试4】不同速度参数测试")
    print("=" * 60)

    print("  目标位置 (笛卡尔):")
    print_pose_list("    ", TARGET_POSE)

    # 测试不同速度组合
    speed_configs = [
        (10, 20, "慢速"),
        (20, 40, "中速"),
        (30, 60, "快速"),
    ]

    for velocity, acc, name in speed_configs:
        print(f"\n  [{name}] 速度={velocity}°/s, 加速度={acc}°/s²")

        robot = RobotController()
        success, msg = robot.connect()
        if not success:
            print(f"    ❌ 连接失败: {msg}")
            continue

        # 读取起始位置
        ok, start_pose, _ = robot.read_current_tcp()
        if not ok:
            print("    ❌ 读取起始位置失败")
            robot.disconnect()
            continue

        print(f"    起始 X: {start_pose['x']:.3f}mm")

        # 执行运动
        success, msg = robot.move_to_cartesian(TARGET_POSE, velocity=velocity, acc=acc)
        if not success:
            print(f"    ❌ {msg}")
            robot.disconnect()
            continue
        print(f"    ✅ 指令已发送")

        # 等待完成
        done, msg = robot.wait_done(timeout=60)
        if not done:
            print(f"    ❌ {msg}")
            robot.disconnect()
            continue
        print(f"    ✅ 运动完成")

        # 读取到达位置
        ok, end_pose, _ = robot.read_current_tcp()
        if not ok:
            print("    ❌ 读取到达位置失败")
            robot.disconnect()
            continue

        # 计算误差
        dx = end_pose['x'] - TARGET_POSE[0]
        dy = end_pose['y'] - TARGET_POSE[1]
        dz = end_pose['z'] - TARGET_POSE[2]
        error_pos = (dx**2 + dy**2 + dz**2) ** 0.5

        print(f"    到达 X: {end_pose['x']:.3f}mm")
        print(f"    位置误差: {error_pos:.3f}mm")
        print(f"    {'✅ 通过' if error_pos < 2.0 else '⚠ 偏差较大'}")

        robot.disconnect()
        time.sleep(1)

    return True


def test_compare_with_joint():
    """测试5: 对比笛卡尔和关节两种方式"""
    print("\n" + "=" * 60)
    print("【测试5】对比笛卡尔坐标 vs 关节坐标")
    print("=" * 60)

    robot = RobotController()
    success, msg = robot.connect()
    if not success:
        print(f"❌ {msg}")
        return False

    # 读取当前位置
    ok, current_joint, msg = robot.read_current_joint()
    if not ok:
        print(f"❌ 读取关节位置失败: {msg}")
        robot.disconnect()
        return False

    print("  当前关节位置:")
    print_joint("    ", current_joint)

    print("\n  [A] 使用笛卡尔坐标 (nIsUseJoint=0)")
    print(f"    目标: X={TARGET_POSE[0]:.1f}, Y={TARGET_POSE[1]:.1f}, Z={TARGET_POSE[2]:.1f}")

    success, msg = robot.move_to_cartesian(TARGET_POSE, velocity=15, acc=30)
    if success:
        robot.wait_done(timeout=30)
        ok, end_pose, _ = robot.read_current_tcp()
        if ok:
            dx = end_pose['x'] - TARGET_POSE[0]
            dy = end_pose['y'] - TARGET_POSE[1]
            dz = end_pose['z'] - TARGET_POSE[2]
            error = (dx**2 + dy**2 + dz**2) ** 0.5
            print(f"    到达误差: {error:.3f}mm")
    else:
        print(f"    ❌ {msg}")

    # 读取当前关节作为目标 (用于B)
    ok, current_joint, _ = robot.read_current_joint()
    if not ok:
        print("❌ 读取关节失败")
        robot.disconnect()
        return False

    print("\n  [B] 使用关节坐标 (nIsUseJoint=1)")
    print(f"    目标: J1={current_joint[0]:.1f}, J2={current_joint[1]:.1f}, J3={current_joint[2]:.1f}")

    ret = robot.cps.HRIF_MoveJ(
        BOX_ID, RBT_ID,
        [0, 0, 0, 0, 0, 0],
        current_joint,
        "TCP_111", "Base",
        15, 30, 0,
        1, 0, 0, 0, "0"
    )
    if ret == 0:
        robot.wait_done(timeout=30)
        print(f"    ✅ 运动成功")
    else:
        print(f"    ❌ 失败, 错误码: {ret}")

    robot.disconnect()
    return True


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("定点运动测试 (笛卡尔坐标 + MoveJ)")
    print(f"机器人IP: {ROBOT_IP}")
    print("=" * 60)

    print("\n目标位置:")
    print_pose_list("  ", TARGET_POSE)

    tests = [
        ("1", "连接机器人", test_connect),
        ("2", "读取当前位置", test_read_current),
        ("3", "运动到目标点", test_move_to_target),
        ("4", "不同速度参数测试", test_move_with_different_speed),
        ("5", "对比笛卡尔 vs 关节", test_compare_with_joint),
    ]

    print("\n可用的测试项:")
    for num, name, _ in tests:
        print(f"  [{num}] {name}")
    print("  [0] 运行所有测试")
    print("  [q] 退出")

    while True:
        choice = input("\n请选择测试项: ").strip()

        if choice.lower() == 'q':
            print("退出")
            break

        if choice == '0':
            print("\n开始运行所有测试...")
            for num, name, func in tests:
                print(f"\n--- 运行 {name} ---")
                func()
                input("\n按 Enter 继续...")
            break

        for num, name, func in tests:
            if choice == num:
                func()
                break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()
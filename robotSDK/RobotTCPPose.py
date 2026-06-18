#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华沿SDK机械臂坐标读取 —— 分步单测
用于手眼标定：只读实际位置
"""

import sys
import os
import time
# ==================== 导入SDK ====================
try:
    from CPS import CPSClient
    print("✅ CPS SDK 导入成功!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


# ==================== 机械臂控制器 ====================
class HuayanRobotController:
    """华沿机械臂控制器 - 用于手眼标定"""

    def __init__(self, robot_ip="192.168.0.10", port=10003, box_id=0, rbt_id=0):
        self.robot_ip = robot_ip
        self.port = port
        self.box_id = box_id
        self.rbt_id = rbt_id
        self.cps = CPSClient()
        self.is_connected = False

    def connect(self):
        """连接机器人（使用 GrpEnable 自动完成上电和使能）"""
        print(f"  连接 {self.robot_ip}:{self.port} ...")
        ret = self.cps.HRIF_Connect(self.box_id, self.robot_ip, self.port)
        if ret != 0:
            return False, f"连接失败, 错误码: {ret}"
        print("  ✓ 控制器已连接")

        # 等待控制器启动
        result = []
        for i in range(30):
            ret = self.cps.HRIF_IsControllerStarted(self.box_id, result)
            if ret == 0 and len(result) > 0 and int(result[0]) == 1:
                print("  ✓ 控制器已启动")
                break
            time.sleep(0.5)
        else:
            return False, "控制器启动超时"

        # 使用 GrpEnable 完成所有初始化
        print("  正在使能 (自动完成电箱连接、上电、使能)...")
        ret = self.cps.HRIF_GrpEnable(self.box_id, self.rbt_id)
        if ret != 0:
            err_result = []
            self.cps.HRIF_GetErrorCodeStr(self.box_id, ret, err_result)
            err_msg = err_result[0] if len(err_result) > 0 else f"错误码 {ret}"
            return False, f"使能失败: {err_msg}"

        print("  ✓ 电箱已连接，已上电，已使能")
        self.is_connected = True
        return True, "连接成功"

    def disconnect(self):
        """断开网络连接（不断电、不去使能）"""
        print("  正在断开网络连接...")
        try:
            self.cps.HRIF_DisConnect(self.box_id)
            print("  ✓ 已断开网络连接")
        except Exception as e:
            print(f"  ⚠ 断开时发生异常: {e}")
        self.is_connected = False

    def read_joint_position(self):
        """
        读取当前实际关节位置 (用于手眼标定)
        返回: J1~J6 角度 [°]
        """
        result = []
        ret = self.cps.HRIF_ReadActPos(self.box_id, self.rbt_id, result)
        if ret != 0 or len(result) < 6:
            return False, None
        return True, {
            "j1": float(result[0]),
            "j2": float(result[1]),
            "j3": float(result[2]),
            "j4": float(result[3]),
            "j5": float(result[4]),
            "j6": float(result[5]),
        }

    def read_tcp_position(self):
        """
        读取当前实际TCP位置 (笛卡尔坐标)
        返回: X, Y, Z [mm], Rx, Ry, Rz [°]
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

    def read_robot_state(self):
        """读取机器人状态（用于诊断）"""
        result = []
        ret = self.cps.HRIF_ReadRobotState(self.box_id, self.rbt_id, result)
        if ret != 0 or len(result) < 13:
            return False, None
        return True, {
            "is_moving": int(result[0]),
            "is_enabled": int(result[1]),
            "has_error": int(result[2]),
            "error_code": int(result[3]),
            "is_emergency": int(result[7]),
            "is_powered": int(result[9]),
            "is_box_connected": int(result[10]),
        }


# ==================== 测试函数 ====================

def get_ip():
    ip = input("请输入机械臂IP (默认 192.168.0.10): ").strip()
    return ip if ip else "192.168.0.10"


def test1_connect():
    """【测试1】连接机器人"""
    print("\n" + "=" * 60)
    print("【测试1】连接机器人")
    print("=" * 60)

    ip = get_ip()
    robot = HuayanRobotController(robot_ip=ip)

    success, msg = robot.connect()
    if success:
        print(f"\n✅ 连接成功!")
        ok, state = robot.read_robot_state()
        if ok:
            print(f"   已使能: {state['is_enabled']} (1=是)")
            print(f"   已上电: {state['is_powered']} (1=是)")
            print(f"   电箱连接: {state['is_box_connected']} (1=是)")
    else:
        print(f"\n❌ 连接失败: {msg}")

    robot.disconnect()
    return success


def test2_read_joint():
    """【测试2】读取关节位置（手眼标定核心数据）"""
    print("\n" + "=" * 60)
    print("【测试2】读取当前关节位置 (用于手眼标定)")
    print("=" * 60)

    ip = get_ip()
    robot = HuayanRobotController(robot_ip=ip)

    success, msg = robot.connect()
    if not success:
        print(f"❌ 连接失败: {msg}")
        return False

    ok, pose = robot.read_joint_position()
    if ok:
        print(f"\n  当前关节位置 (baseTend):")
        print(f"    J1 = {pose['j1']:10.3f}°")
        print(f"    J2 = {pose['j2']:10.3f}°")
        print(f"    J3 = {pose['j3']:10.3f}°")
        print(f"    J4 = {pose['j4']:10.3f}°")
        print(f"    J5 = {pose['j5']:10.3f}°")
        print(f"    J6 = {pose['j6']:10.3f}°")
    else:
        print("❌ 读取失败")

    robot.disconnect()
    return ok


def test3_read_tcp():
    """【测试3】读取TCP笛卡尔位置"""
    print("\n" + "=" * 60)
    print("【测试3】读取当前TCP笛卡尔位置")
    print("=" * 60)

    ip = get_ip()
    robot = HuayanRobotController(robot_ip=ip)

    success, msg = robot.connect()
    if not success:
        print(f"❌ 连接失败: {msg}")
        return False

    ok, tcp = robot.read_tcp_position()
    if ok:
        print(f"\n  当前TCP位置:")
        print(f"    X  = {tcp['x']:10.3f} mm")
        print(f"    Y  = {tcp['y']:10.3f} mm")
        print(f"    Z  = {tcp['z']:10.3f} mm")
        print(f"    Rx = {tcp['rx']:10.3f}°")
        print(f"    Ry = {tcp['ry']:10.3f}°")
        print(f"    Rz = {tcp['rz']:10.3f}°")
    else:
        print("❌ 读取失败")

    robot.disconnect()
    return ok


def test4_state():
    """【测试4】读取机器人状态（诊断用）"""
    print("\n" + "=" * 60)
    print("【测试4】读取机器人状态")
    print("=" * 60)

    ip = get_ip()
    robot = HuayanRobotController(robot_ip=ip)

    success, msg = robot.connect()
    if not success:
        print(f"❌ 连接失败: {msg}")
        return False

    ok, state = robot.read_robot_state()
    if ok:
        print("\n  机器人状态:")
        print(f"    运动中:      {state['is_moving']} (1=是)")
        print(f"    已使能:      {state['is_enabled']} (1=是)")
        print(f"    有错误:      {state['has_error']} (1=是)")
        print(f"    错误码:      {state['error_code']}")
        print(f"    已上电:      {state['is_powered']} (1=是)")
        print(f"    电箱连接:    {state['is_box_connected']} (1=是)")
        print(f"    急停:        {state['is_emergency']} (1=是) ← 必须为0")
    else:
        print("❌ 读取失败")

    robot.disconnect()
    return ok


def test5_multi_read():
    """【测试5】连续读取稳定性"""
    print("\n" + "=" * 60)
    print("【测试5】连续读取 10 次关节位置")
    print("=" * 60)

    ip = get_ip()
    count = 10
    robot = HuayanRobotController(robot_ip=ip)

    success, msg = robot.connect()
    if not success:
        print(f"❌ 连接失败: {msg}")
        return False

    print(f"\n连续读取 {count} 次...")
    success_count = 0
    j1_values = []

    for i in range(count):
        ok, pose = robot.read_joint_position()
        if ok:
            success_count += 1
            j1_values.append(pose["j1"])
            print(f"  [{i + 1:2d}] ✓ J1={pose['j1']:8.3f}°  J2={pose['j2']:8.3f}°  J3={pose['j3']:8.3f}°")
        else:
            print(f"  [{i + 1:2d}] ✗ 读取失败")
        time.sleep(0.2)

    print(f"\n统计: 成功 {success_count}/{count} 次")

    if len(j1_values) > 1:
        import statistics
        stdev = statistics.stdev(j1_values)
        print(f"  J1 标准差: {stdev:.6f}° (越小越稳定)")

    robot.disconnect()
    return success_count == count


# ==================== 主菜单 ====================

def main():
    print("=" * 60)
    print("华沿SDK机械臂坐标读取 —— 分步单测")
    print("用于手眼标定")
    print("=" * 60)

    tests = [
        ("1", "连接机器人", test1_connect),
        ("2", "读取关节位置 (手眼标定核心)", test2_read_joint),
        ("3", "读取TCP笛卡尔位置", test3_read_tcp),
        ("4", "读取机器人状态 (诊断)", test4_state),
        ("5", "连续读取稳定性", test5_multi_read),
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华沿机器人 TCP 工具坐标单测
测试功能:
1. 读取当前TCP
2. 设置临时TCP
3. 按名称读取TCP
4. 按名称切换TCP
5. 读取TCP列表
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


# ==================== 工具函数 ====================

def print_tcp(name: str, tcp: list):
    """格式化打印TCP信息"""
    if tcp is None or len(tcp) < 6:
        print(f"  {name}: 无效数据")
        return
    print(f"  {name}:")
    print(f"    X  = {tcp[0]:8.3f} mm")
    print(f"    Y  = {tcp[1]:8.3f} mm")
    print(f"    Z  = {tcp[2]:8.3f} mm")
    print(f"    Rx = {tcp[3]:8.3f}°")
    print(f"    Ry = {tcp[4]:8.3f}°")
    print(f"    Rz = {tcp[5]:8.3f}°")


def connect_robot() -> Tuple[bool, CPSClient]:
    """连接机器人并初始化"""
    cps = CPSClient()

    print("  连接控制器...")
    ret = cps.HRIF_Connect(BOX_ID, ROBOT_IP, PORT)
    if ret != 0:
        return False, f"连接失败: {ret}"
    print("  ✓ 控制器已连接")

    # 等待启动
    result = []
    for i in range(30):
        ret = cps.HRIF_IsControllerStarted(BOX_ID, result)
        if ret == 0 and len(result) > 0 and int(result[0]) == 1:
            print("  ✓ 控制器已启动")
            break
        time.sleep(0.5)
    else:
        return False, "控制器启动超时"

    # 使用 GrpEnable 自动完成电箱连接、上电、使能
    print("  正在使能...")
    ret = cps.HRIF_GrpEnable(BOX_ID, RBT_ID)
    if ret != 0:
        return False, f"使能失败, 错误码: {ret}"
    print("  ✓ 已使能")

    return True, cps


def disconnect_robot(cps: CPSClient):
    """断开连接（不断电，只断开网络）"""
    try:
        cps.HRIF_DisConnect(BOX_ID)
        print("  ✓ 已断开连接")
    except:
        pass


# ==================== TCP单测类 ====================

class TCPTest:
    """TCP工具坐标测试"""

    def __init__(self, cps: CPSClient):
        self.cps = cps
        self.box_id = BOX_ID
        self.rbt_id = RBT_ID

    # ==================== 测试1: 读取当前TCP ====================

    def test_read_current_tcp(self) -> bool:
        """
        测试1: 读取当前工具坐标 (HRIF_ReadCurTCP)
        """
        print("\n" + "=" * 60)
        print("【测试1】读取当前工具坐标 (TCP)")
        print("=" * 60)

        result = []
        ret = self.cps.HRIF_ReadCurTCP(self.box_id, self.rbt_id, result)

        if ret != 0:
            print(f"❌ 读取失败, 错误码: {ret}")
            # 获取错误描述
            err_result = []
            self.cps.HRIF_GetErrorCodeStr(self.box_id, ret, err_result)
            if len(err_result) > 0:
                print(f"   错误描述: {err_result[0]}")
            return False

        if len(result) < 6:
            print(f"❌ 返回数据不足: {len(result)}")
            return False

        tcp = [float(x) for x in result[:6]]
        print("✅ 当前工具坐标:")
        print_tcp("TCP", tcp)

        # 判断是否为默认TCP (全0)
        is_default = all(abs(x) < 0.001 for x in tcp)
        if is_default:
            print("  ℹ 当前TCP为默认值 (全0)，表示工具中心在法兰盘中心")
        else:
            print("  ℹ 当前TCP已设置自定义值")

        return True

    # ==================== 测试2: 设置临时TCP ====================

    def test_set_temporary_tcp(self) -> bool:
        """
        测试2: 设置临时工具坐标 (HRIF_SetTCP)
        注意: 此设置重启后失效，不写入配置文件
        """
        print("\n" + "=" * 60)
        print("【测试2】设置临时工具坐标 (重启后失效)")
        print("=" * 60)

        # 读取当前TCP
        result = []
        ret = self.cps.HRIF_ReadCurTCP(self.box_id, self.rbt_id, result)
        if ret == 0 and len(result) >= 6:
            old_tcp = [float(x) for x in result[:6]]
            print("  当前TCP:")
            print_tcp("  旧值", old_tcp)
        else:
            old_tcp = None
            print("  当前TCP: 读取失败")

        # 设置一个临时TCP (偏移量: X=10, Y=5, Z=20)
        print("\n  设置临时TCP: X=10, Y=5, Z=20, Rx=0, Ry=0, Rz=0")
        new_tcp = [10.0, 5.0, 20.0, 0.0, 0.0, 0.0]
        ret = self.cps.HRIF_SetTCP(self.box_id, self.rbt_id, new_tcp)

        if ret != 0:
            print(f"❌ 设置失败, 错误码: {ret}")
            return False

        print("  ✅ 设置成功")

        # 验证是否设置成功
        time.sleep(0.5)
        result = []
        ret = self.cps.HRIF_ReadCurTCP(self.box_id, self.rbt_id, result)
        if ret == 0 and len(result) >= 6:
            verify_tcp = [float(x) for x in result[:6]]
            print("  验证当前TCP:")
            print_tcp("  新值", verify_tcp)

            # 检查是否设置正确
            if all(abs(verify_tcp[i] - new_tcp[i]) < 0.001 for i in range(6)):
                print("  ✅ 验证通过: TCP已正确设置")
            else:
                print("  ⚠ 验证失败: TCP值不匹配")
                return False
        else:
            print("  ❌ 验证失败: 无法读取TCP")
            return False

        # 恢复TCP为0 (可选)
        print("\n  恢复TCP为默认值 (全0)")
        ret = self.cps.HRIF_SetTCP(self.box_id, self.rbt_id, [0, 0, 0, 0, 0, 0])
        if ret == 0:
            print("  ✅ 已恢复")
        else:
            print(f"  ⚠ 恢复失败, 错误码: {ret}")

        return True

    # ==================== 测试3: 按名称读取TCP ====================

    def test_read_tcp_by_name(self, tcp_name: str = "TCP") -> bool:
        """
        测试3: 按名称读取工具坐标 (HRIF_ReadTCPByName)
        """
        print("\n" + "=" * 60)
        print(f"【测试3】按名称读取工具坐标: '{tcp_name}'")
        print("=" * 60)

        result = []
        ret = self.cps.HRIF_ReadTCPByName(self.box_id, self.rbt_id, tcp_name, result)

        if ret != 0:
            print(f"❌ 读取失败, 错误码: {ret}")
            # 错误码20006表示"未找到指定对象"
            if ret == 20006:
                print(f"   提示: 名为 '{tcp_name}' 的TCP不存在")
            return False

        if len(result) < 6:
            print(f"❌ 返回数据不足: {len(result)}")
            return False

        tcp = [float(x) for x in result[:6]]
        print(f"✅ TCP '{tcp_name}' 的坐标:")
        print_tcp("", tcp)
        return True

    # ==================== 测试4: 按名称切换TCP ====================

    def test_switch_tcp_by_name(self, tcp_name: str = "TCP") -> bool:
        """
        测试4: 按名称切换工具坐标 (HRIF_SetTCPByName)
        """
        print("\n" + "=" * 60)
        print(f"【测试4】按名称切换工具坐标: '{tcp_name}'")
        print("=" * 60)

        # 读取当前TCP
        result = []
        ret = self.cps.HRIF_ReadCurTCP(self.box_id, self.rbt_id, result)
        if ret == 0 and len(result) >= 6:
            old_tcp = [float(x) for x in result[:6]]
            print("  切换前:")
            print_tcp("  当前TCP", old_tcp)
        else:
            print("  切换前: 读取当前TCP失败")

        # 切换TCP
        print(f"\n  切换到: '{tcp_name}'")
        ret = self.cps.HRIF_SetTCPByName(self.box_id, self.rbt_id, tcp_name)

        if ret != 0:
            print(f"❌ 切换失败, 错误码: {ret}")
            if ret == 20006:
                print(f"   提示: 名为 '{tcp_name}' 的TCP不存在")
            return False

        print("  ✅ 切换成功")

        # 验证切换结果
        time.sleep(0.5)
        result = []
        ret = self.cps.HRIF_ReadCurTCP(self.box_id, self.rbt_id, result)
        if ret == 0 and len(result) >= 6:
            new_tcp = [float(x) for x in result[:6]]
            print("\n  切换后:")
            print_tcp("  当前TCP", new_tcp)
        else:
            print("  ⚠ 无法验证切换结果")

        return True

    # ==================== 测试5: 读取TCP列表 ====================

    def test_read_tcp_list(self) -> bool:
        """
        测试5: 读取系统中所有TCP名称 (HRIF_ReadTCPList)
        """
        print("\n" + "=" * 60)
        print("【测试5】读取所有TCP名称列表")
        print("=" * 60)

        result = []
        ret = self.cps.HRIF_ReadTCPList(self.box_id, self.rbt_id, result)

        if ret != 0:
            print(f"❌ 读取失败, 错误码: {ret}")
            return False

        if len(result) == 0:
            print("ℹ 系统中没有任何TCP定义")
            return True

        print(f"✅ 共 {len(result)} 个TCP:")
        for i, name in enumerate(result):
            print(f"  [{i + 1}] {name}")

        # 打印每个TCP的详细信息
        print("\n  详细信息:")
        for name in result:
            tcp_result = []
            ret = self.cps.HRIF_ReadTCPByName(self.box_id, self.rbt_id, name, tcp_result)
            if ret == 0 and len(tcp_result) >= 6:
                tcp = [float(x) for x in tcp_result[:6]]
                print(f"    {name}: X={tcp[0]:.2f}, Y={tcp[1]:.2f}, Z={tcp[2]:.2f}")
            else:
                print(f"    {name}: 读取详情失败")

        return True

    # ==================== 测试6: 完整TCP操作流程 ====================

    def test_full_tcp_workflow(self) -> bool:
        """
        测试6: 完整TCP操作流程
        1. 读取当前TCP
        2. 查看所有TCP
        3. 切换到一个TCP
        4. 验证切换结果
        """
        print("\n" + "=" * 60)
        print("【测试6】完整TCP操作流程")
        print("=" * 60)

        # 1. 读取所有TCP
        result = []
        ret = self.cps.HRIF_ReadTCPList(self.box_id, self.rbt_id, result)
        if ret != 0:
            print(f"❌ 读取TCP列表失败: {ret}")
            return False

        if len(result) == 0:
            print("ℹ 系统中无TCP定义，请先在示教器中创建TCP")
            return True

        print(f"✅ 可用TCP列表: {result}")

        # 2. 尝试切换到一个存在的TCP
        target_tcp = result[0]  # 使用第一个
        print(f"\n  尝试切换到: '{target_tcp}'")
        ret = self.cps.HRIF_SetTCPByName(self.box_id, self.rbt_id, target_tcp)

        if ret != 0:
            print(f"❌ 切换失败: {ret}")
            return False

        print("  ✅ 切换成功")

        # 3. 验证
        time.sleep(0.5)
        tcp_result = []
        ret = self.cps.HRIF_ReadCurTCP(self.box_id, self.rbt_id, tcp_result)
        if ret == 0 and len(tcp_result) >= 6:
            current_tcp = [float(x) for x in tcp_result[:6]]
            print("  切换后的TCP:")
            print_tcp("", current_tcp)

            # 读取该TCP的定义值，对比是否一致
            detail_result = []
            ret = self.cps.HRIF_ReadTCPByName(self.box_id, self.rbt_id, target_tcp, detail_result)
            if ret == 0 and len(detail_result) >= 6:
                defined_tcp = [float(x) for x in detail_result[:6]]
                print(f"\n  TCP '{target_tcp}' 的定义值:")
                print_tcp("", defined_tcp)

                # 检查是否一致
                if all(abs(current_tcp[i] - defined_tcp[i]) < 0.01 for i in range(6)):
                    print("\n  ✅ 验证通过: 切换后的TCP与定义值一致")
                else:
                    print("\n  ⚠ 警告: 切换后的TCP与定义值不一致")
                    return False
            else:
                print(f"  ⚠ 无法读取TCP '{target_tcp}' 的定义值")
        else:
            print("  ❌ 无法读取当前TCP")
            return False

        return True


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("华沿机器人 TCP 工具坐标单测")
    print(f"机器人IP: {ROBOT_IP}")
    print("=" * 60)

    # 连接机器人
    print("\n[连接] 正在连接机器人...")
    print("-" * 40)
    success, cps = connect_robot()
    if not success:
        print(f"❌ 连接失败: {cps}")
        return
    print("✅ 连接成功")

    tcp_test = TCPTest(cps)

    try:
        # 显示菜单
        tests = [
            ("1", "读取当前TCP", tcp_test.test_read_current_tcp),
            ("2", "设置临时TCP", tcp_test.test_set_temporary_tcp),
            ("3", "按名称读取TCP", lambda: tcp_test.test_read_tcp_by_name("TCP")),
            ("4", "按名称切换TCP", lambda: tcp_test.test_switch_tcp_by_name("TCP")),
            ("5", "读取TCP列表", tcp_test.test_read_tcp_list),
            ("6", "完整TCP操作流程", tcp_test.test_full_tcp_workflow),
        ]

        print("\n" + "=" * 60)
        print("可用的测试项:")
        for num, name, _ in tests:
            print(f"  [{num}] {name}")
        print("  [0] 运行所有测试")
        print("  [q] 退出")
        print("=" * 60)

        while True:
            choice = input("\n请选择测试项: ").strip()

            if choice.lower() == 'q':
                print("退出")
                break

            if choice == '0':
                print("\n开始运行所有测试...\n")
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
                print("无效选择，请重新输入")

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
    except Exception as e:
        print(f"\n❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        disconnect_robot(cps)


if __name__ == "__main__":
    main()
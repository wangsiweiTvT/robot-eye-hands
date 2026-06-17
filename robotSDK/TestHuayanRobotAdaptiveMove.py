import unittest
import time
from CPS import CPSClient


class TestHuayanRobotAdaptiveMove(unittest.TestCase):
    """基于安全限制的自适应运动测试"""

    @classmethod
    def setUpClass(cls):
        """初始化并读取安全限制"""
        cls.cps = CPSClient()
        cls.box_id = 0
        cls.rbt_id = 0
        cls.ip = '192.168.0.10'
        cls.port = 10003

        print("\n" + "=" * 70)
        print("【初始化机器人并读取安全限制】")
        print("=" * 70)

        # 连接
        nRet = cls.cps.HRIF_Connect(cls.box_id, cls.ip, cls.port)
        print(f"连接结果: {nRet}")

        if nRet != 0:
            print("⚠️ 连接失败，请检查IP地址")
            return

        # 使能
        nRet = cls.cps.HRIF_GrpEnable(cls.box_id, cls.rbt_id)
        print(f"使能结果: {nRet}")
        time.sleep(2)

        # 读取所有安全限制
        cls._read_limits()

        # 计算安全运动参数
        cls._calculate_safe_params()

        print("=" * 70 + "\n")

    @classmethod
    def tearDownClass(cls):
        """清理资源"""
        print("\n" + "=" * 70)
        print("清理资源...")
        try:
            cls.cps.HRIF_GrpDisable(cls.box_id, cls.rbt_id)
        except:
            pass
        try:
            cls.cps.HRIF_DisConnect(cls.box_id)
        except:
            pass
        print("=" * 70)

    @classmethod
    def _read_limits(cls):
        """读取所有限制参数"""
        print("\n--- 读取安全限制 ---")

        # 1. 速度比
        result = []
        cls.cps.HRIF_ReadOverride(cls.box_id, cls.rbt_id, result)
        cls.override = float(result[0]) if result else 1.0
        print(f"  速度比: {cls.override * 100:.0f}%")

        # 2. 关节最大速度
        result = []
        cls.cps.HRIF_ReadJointMaxVel(cls.box_id, cls.rbt_id, result)
        if result:
            cls.joint_max_vel = [float(x) for x in result]
            print(f"  关节最大速度: {cls.joint_max_vel}")
        else:
            cls.joint_max_vel = [180] * 6
            print(f"  关节最大速度: 使用默认值 {cls.joint_max_vel}")

        # 3. 关节最大加速度
        result = []
        cls.cps.HRIF_ReadJointMaxAcc(cls.box_id, cls.rbt_id, result)
        if result:
            cls.joint_max_acc = [float(x) for x in result]
            print(f"  关节最大加速度: {cls.joint_max_acc}")
        else:
            cls.joint_max_acc = [360] * 6
            print(f"  关节最大加速度: 使用默认值 {cls.joint_max_acc}")

        # 4. 直线最大速度
        result = []
        cls.cps.HRIF_ReadLinearMaxSpeed(cls.box_id, cls.rbt_id, result)
        if result and len(result) >= 3:
            cls.linear_max_vel = float(result[0])
            cls.linear_max_acc = float(result[1])
            print(f"  直线最大速度: {cls.linear_max_vel} mm/s")
            print(f"  直线最大加速度: {cls.linear_max_acc} mm/s²")
        else:
            cls.linear_max_vel = 2000
            cls.linear_max_acc = 2500
            print(f"  直线最大速度: 使用默认值 {cls.linear_max_vel} mm/s")
            print(f"  直线最大加速度: 使用默认值 {cls.linear_max_acc} mm/s²")

        # 5. 最大负载
        result = []
        cls.cps.HRIF_ReadMaxPayload(cls.box_id, result)
        cls.max_payload = float(result[0]) if result else 6.0
        print(f"  最大负载: {cls.max_payload} kg")

        # 6. 急停状态 - ✅ 修复：只需要 box_id 和 result，不需要 rbt_id
        result = []
        try:
            cls.cps.HRIF_ReadEmergencyInfo(cls.box_id, result)
            if result and len(result) >= 4:
                print(f"  急停回路错误: {'是' if result[0] == '1' else '否'}")
                print(f"  急停信号: {'触发' if result[1] == '1' else '正常'}")
                print(f"  安全光幕错误: {'是' if result[2] == '1' else '否'}")
                print(f"  安全光幕信号: {'触发' if result[3] == '1' else '正常'}")
        except Exception as e:
            print(f"  读取急停信息失败: {e}")

        # 7. 机器人状态
        result = []
        cls.cps.HRIF_ReadRobotState(cls.box_id, cls.rbt_id, result)
        if result and len(result) >= 13:
            print(f"  机器人状态: {'已使能' if result[1] == '1' else '未使能'}")
            print(f"  错误状态: {'有错误' if result[2] == '1' else '正常'}")
            print(f"  错误码: {result[3]}")
            print(f"  抱闸状态: {'抱闸' if result[5] == '0' else '松闸'}")
            print(f"  暂停状态: {'暂停' if result[6] == '1' else '运行中'}")
            print(f"  到位状态: {'到位' if result[12] == '1' else '运动中'}")
        else:
            print(f"  读取机器人状态失败, 错误码: {nRet if 'nRet' in dir() else '未知'}")

    @classmethod
    def _calculate_safe_params(cls):
        """根据安全限制计算安全运动参数"""
        print("\n--- 计算安全运动参数 ---")

        # 获取关节速度最小值
        min_joint_vel = min(cls.joint_max_vel) if cls.joint_max_vel else 180

        # 关节运动参数：使用最大值的 30% ~ 70%
        # 加速度必须大于速度

        # 低速（30% 最大速度）
        cls.joint_slow_vel = max(10, min(min_joint_vel * 0.3, 60))
        cls.joint_slow_acc = max(20, cls.joint_slow_vel * 1.5)

        # 中速（50% 最大速度）
        cls.joint_medium_vel = max(20, min(min_joint_vel * 0.5, 100))
        cls.joint_medium_acc = max(30, cls.joint_medium_vel * 1.5)

        # 快速（70% 最大速度）
        cls.joint_fast_vel = max(30, min(min_joint_vel * 0.7, 140))
        cls.joint_fast_acc = max(50, cls.joint_fast_vel * 1.5)

        # 直线运动参数
        cls.linear_slow_vel = max(50, min(cls.linear_max_vel * 0.2, 300))
        cls.linear_slow_acc = max(100, cls.linear_slow_vel * 1.5)

        cls.linear_medium_vel = max(100, min(cls.linear_max_vel * 0.4, 500))
        cls.linear_medium_acc = max(200, cls.linear_medium_vel * 1.5)

        cls.linear_fast_vel = max(150, min(cls.linear_max_vel * 0.6, 800))
        cls.linear_fast_acc = max(300, cls.linear_fast_vel * 1.5)

        print(f"  关节低速: {cls.joint_slow_vel:.0f}°/s, 加速度: {cls.joint_slow_acc:.0f}°/s²")
        print(f"  关节中速: {cls.joint_medium_vel:.0f}°/s, 加速度: {cls.joint_medium_acc:.0f}°/s²")
        print(f"  关节快速: {cls.joint_fast_vel:.0f}°/s, 加速度: {cls.joint_fast_acc:.0f}°/s²")
        print(f"  直线低速: {cls.linear_slow_vel:.0f}mm/s, 加速度: {cls.linear_slow_acc:.0f}mm/s²")
        print(f"  直线中速: {cls.linear_medium_vel:.0f}mm/s, 加速度: {cls.linear_medium_acc:.0f}mm/s²")
        print(f"  直线快速: {cls.linear_fast_vel:.0f}mm/s, 加速度: {cls.linear_fast_acc:.0f}mm/s²")

    def _read_position(self):
        """读取当前位置"""
        result = []
        nRet = self.cps.HRIF_ReadActPos(self.box_id, self.rbt_id, result)
        if nRet == 0 and len(result) >= 12:
            print(f"  关节: J1={float(result[0]):.1f}°, J2={float(result[1]):.1f}°, J3={float(result[2]):.1f}°, "
                  f"J4={float(result[3]):.1f}°, J5={float(result[4]):.1f}°, J6={float(result[5]):.1f}°")
            print(f"  笛卡尔: X={float(result[6]):.1f}mm, Y={float(result[7]):.1f}mm, Z={float(result[8]):.1f}mm")
            return result
        return []

    def _wait_motion_done(self, timeout=30):
        """等待运动完成"""
        start = time.time()
        while True:
            result = []
            self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
            if result and result[0] == 'True':
                elapsed = time.time() - start
                print(f"  ✅ 运动完成 (耗时: {elapsed:.2f}s)")
                return True
            if time.time() - start > timeout:
                print("  ❌ 运动超时!")
                return False
            time.sleep(0.05)

    def _execute_move(self, move_func, *args, **kwargs):
        """执行运动并等待完成"""
        nRet = move_func(*args, **kwargs)
        print(f"  指令返回: {nRet}")
        if nRet == 0:
            self._wait_motion_done()
            self._read_position()
            return True
        else:
            err = []
            self.cps.HRIF_GetErrorCodeStr(self.box_id, nRet, err)
            print(f"  ❌ 错误: {err[0] if err else '未知'}")
            return False

    # ============================================================
    # 测试用例
    # ============================================================

    def test_01_joint_slow(self):
        """低速关节运动"""
        print("\n" + "=" * 70)
        print(f"【测试1】低速关节运动 (速度: {self.joint_slow_vel:.0f}°/s)")
        print("=" * 70)

        point = [0, 0, 90, 0, 90, 0]
        raw_acs = [0, 0, 90, 0, 90, 0]

        success = self._execute_move(
            self.cps.HRIF_MoveJ,
            self.box_id, self.rbt_id,
            point, raw_acs, "TCP", "Base",
            self.joint_slow_vel, self.joint_slow_acc, 0,
            1, 0, 0, 0, "1"
        )
        self.assertTrue(success)

    def test_02_joint_medium(self):
        """中速关节运动"""
        print("\n" + "=" * 70)
        print(f"【测试2】中速关节运动 (速度: {self.joint_medium_vel:.0f}°/s)")
        print("=" * 70)

        point = [10, -10, 80, 0, 80, 0]
        raw_acs = [10, -10, 80, 0, 80, 0]

        success = self._execute_move(
            self.cps.HRIF_MoveJ,
            self.box_id, self.rbt_id,
            point, raw_acs, "TCP", "Base",
            self.joint_medium_vel, self.joint_medium_acc, 0,
            1, 0, 0, 0, "2"
        )
        self.assertTrue(success)

    def test_03_joint_fast(self):
        """快速关节运动"""
        print("\n" + "=" * 70)
        print(f"【测试3】快速关节运动 (速度: {self.joint_fast_vel:.0f}°/s)")
        print("=" * 70)

        point = [0, 0, 90, 0, 90, 0]
        raw_acs = [0, 0, 90, 0, 90, 0]

        success = self._execute_move(
            self.cps.HRIF_MoveJ,
            self.box_id, self.rbt_id,
            point, raw_acs, "TCP", "Base",
            self.joint_fast_vel, self.joint_fast_acc, 0,
            1, 0, 0, 0, "3"
        )
        self.assertTrue(success)

    def test_04_linear_slow(self):
        """低速直线运动"""
        print("\n" + "=" * 70)
        print(f"【测试4】低速直线运动 (速度: {self.linear_slow_vel:.0f}mm/s)")
        print("=" * 70)

        current = self._read_position()
        if not current:
            self.skipTest("无法读取当前位置")

        raw_acs = [float(current[0]), float(current[1]), float(current[2]),
                   float(current[3]), float(current[4]), float(current[5])]

        point = [420, 0, 445, 180, 0, 180]

        success = self._execute_move(
            self.cps.HRIF_MoveL,
            self.box_id, self.rbt_id,
            point, raw_acs, "TCP", "Base",
            self.linear_slow_vel, self.linear_slow_acc, 0,
            0, 0, 0, "4"
        )
        self.assertTrue(success)

    def test_05_linear_medium(self):
        """中速直线运动"""
        print("\n" + "=" * 70)
        print(f"【测试5】中速直线运动 (速度: {self.linear_medium_vel:.0f}mm/s)")
        print("=" * 70)

        current = self._read_position()
        if not current:
            self.skipTest("无法读取当前位置")

        raw_acs = [float(current[0]), float(current[1]), float(current[2]),
                   float(current[3]), float(current[4]), float(current[5])]

        point = [380, 50, 445, 180, 0, 180]

        success = self._execute_move(
            self.cps.HRIF_MoveL,
            self.box_id, self.rbt_id,
            point, raw_acs, "TCP", "Base",
            self.linear_medium_vel, self.linear_medium_acc, 0,
            0, 0, 0, "5"
        )
        self.assertTrue(success)

    def test_06_linear_fast(self):
        """快速直线运动"""
        print("\n" + "=" * 70)
        print(f"【测试6】快速直线运动 (速度: {self.linear_fast_vel:.0f}mm/s)")
        print("=" * 70)

        current = self._read_position()
        if not current:
            self.skipTest("无法读取当前位置")

        raw_acs = [float(current[0]), float(current[1]), float(current[2]),
                   float(current[3]), float(current[4]), float(current[5])]

        point = [420, 0, 445, 180, 0, 180]

        success = self._execute_move(
            self.cps.HRIF_MoveL,
            self.box_id, self.rbt_id,
            point, raw_acs, "TCP", "Base",
            self.linear_fast_vel, self.linear_fast_acc, 0,
            0, 0, 0, "6"
        )
        self.assertTrue(success)

    def test_07_waypoint_chain(self):
        """多点连续路点运动 (带过渡)"""
        print("\n" + "=" * 70)
        print(f"【测试7】多点连续路点运动 (速度: {self.joint_medium_vel:.0f}°/s, 过渡半径: 10mm)")
        print("=" * 70)

        waypoints = [
            [0, 0, 90, 0, 90, 0],
            [15, -10, 80, 0, 80, 0],
            [30, -20, 70, 0, 70, 0],
            [15, -10, 80, 0, 80, 0],
            [0, 0, 90, 0, 90, 0],
        ]

        for i, wp in enumerate(waypoints):
            print(f"\n  移动到点 {i + 1}: {wp}")
            nRet = self.cps.HRIF_MoveJ(
                self.box_id, self.rbt_id,
                wp, wp, "TCP", "Base",
                self.joint_medium_vel, self.joint_medium_acc, 10,
                1, 0, 0, 0, str(i + 1)
            )
            print(f"  指令返回: {nRet}")
            if nRet == 0:
                self._wait_motion_done(timeout=15)
            else:
                err = []
                self.cps.HRIF_GetErrorCodeStr(self.box_id, nRet, err)
                print(f"  ❌ 错误: {err[0] if err else '未知'}")
                self.fail(f"路点 {i + 1} 运动失败")

        self._read_position()
        print("\n  ✅ 所有路点执行完成")

    def test_08_relative_motion(self):
        """相对运动测试"""
        print("\n" + "=" * 70)
        print("【测试8】相对运动 (关节J1正方向移动5°)")
        print("=" * 70)

        nRet = self.cps.HRIF_MoveRelJ(
            self.box_id, self.rbt_id,
            0,  # J1轴
            1,  # 正方向
            5.0  # 5度
        )
        print(f"  指令返回: {nRet}")
        if nRet == 0:
            self._wait_motion_done()
            self._read_position()
        else:
            err = []
            self.cps.HRIF_GetErrorCodeStr(self.box_id, nRet, err)
            print(f"  ❌ 错误: {err[0] if err else '未知'}")
            self.fail("相对运动失败")

    def test_09_move_to_safe(self):
        """移动到安全位置"""
        print("\n" + "=" * 70)
        print("【测试9】移动到安全位置 (MoveToSS)")
        print("=" * 70)

        nRet = self.cps.HRIF_MoveToSS(self.box_id)
        print(f"  指令返回: {nRet}")
        if nRet == 0:
            self._wait_motion_done()
            self._read_position()
        else:
            err = []
            self.cps.HRIF_GetErrorCodeStr(self.box_id, nRet, err)
            print(f"  ❌ 错误: {err[0] if err else '未知'}")
            self.fail("移动到安全位置失败")

    def test_10_performance_test(self):
        """性能测试 - 记录运动时间"""
        print("\n" + "=" * 70)
        print("【测试10】性能测试 - 不同速度下的运动时间")
        print("=" * 70)

        # 先回到零点
        print("\n  准备: 回到零点位置")
        self.cps.HRIF_MoveJ(
            self.box_id, self.rbt_id,
            [0, 0, 0, 0, 0, 0],
            [0, 0, 90, 0, 90, 0],
            "TCP", "Base",
            50, 80, 0,
            1, 0, 0, 0, "start"
        )
        self._wait_motion_done()

        test_speeds = [
            (30, 50, "低速"),
            (60, 100, "中速"),
            (self.joint_fast_vel, self.joint_fast_acc, "高速"),
        ]

        print("\n  ┌────┬──────────┬──────────┬──────────┐")
        print("  │    │   速度   │  加速度  │   耗时   │")
        print("  ├────┼──────────┼──────────┼──────────┤")

        for vel, acc, name in test_speeds:
            start = time.time()
            nRet = self.cps.HRIF_MoveJ(
                self.box_id, self.rbt_id,
                [10, -5, 85, 0, 85, 0],
                [10, -5, 85, 0, 85, 0],
                "TCP", "Base",
                vel, acc, 0,
                1, 0, 0, 0, "perf"
            )
            if nRet == 0:
                self._wait_motion_done()
                elapsed = time.time() - start
                print(f"  │ {name:2s} │ {vel:>8.0f}°/s │ {acc:>8.0f}°/s² │ {elapsed:>8.2f}s │")
            else:
                print(f"  │ {name:2s} │ {vel:>8.0f}°/s │ {acc:>8.0f}°/s² │   失败    │")

        print("  └────┴──────────┴──────────┴──────────┘")


if __name__ == '__main__':
    unittest.main(verbosity=2)
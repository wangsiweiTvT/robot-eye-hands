import unittest
import time
import math
from CPS import CPSClient


class TestHuayanRobotMoveToPosition(unittest.TestCase):
    """运动到指定位置测试"""

    @classmethod
    def setUpClass(cls):
        """初始化连接并使能机器人"""
        cls.cps = CPSClient()
        cls.box_id = 0
        cls.rbt_id = 0
        cls.ip = '192.168.0.10'
        cls.port = 10003

        print("\n" + "=" * 60)
        print("开始初始化机器人连接...")

        # 连接机器人
        nRet = cls.cps.HRIF_Connect(cls.box_id, cls.ip, cls.port)
        print(f"连接结果: {nRet}")

        if nRet != 0:
            print("连接失败，请检查IP地址和网络")
            return

        # 检查连接状态
        is_connected = cls.cps.HRIF_IsConnected(cls.box_id)
        print(f"连接状态: {is_connected}")

        # 使能机器人
        print("正在使能机器人...")
        nRet = cls.cps.HRIF_GrpEnable(cls.box_id, cls.rbt_id)
        print(f"使能结果: {nRet}")

        if nRet != 0:
            print("使能失败，请检查机器人状态")

        # 等待使能完成
        time.sleep(2)

        # 读取初始位置
        cls._read_current_position(cls)
        print("=" * 60 + "\n")

    @classmethod
    def tearDownClass(cls):
        """清理：去使能并断开连接"""
        print("\n" + "=" * 60)
        print("开始清理资源...")

        try:
            # 去使能
            cls.cps.HRIF_GrpDisable(cls.box_id, cls.rbt_id)
            print("已去使能")
        except Exception as e:
            print(f"去使能异常: {e}")

        try:
            # 断开连接
            cls.cps.HRIF_DisConnect(cls.box_id)
            print("已断开连接")
        except Exception as e:
            print(f"断开连接异常: {e}")

        print("=" * 60)

    def _read_current_position(self):
        """读取并打印当前位置"""
        result = []
        nRet = self.cps.HRIF_ReadActPos(self.box_id, self.rbt_id, result)
        if nRet == 0 and len(result) >= 12:
            print("\n--- 当前位置 ---")
            print(f"关节: J1={float(result[0]):.2f}°, J2={float(result[1]):.2f}°, J3={float(result[2]):.2f}°, "
                  f"J4={float(result[3]):.2f}°, J5={float(result[4]):.2f}°, J6={float(result[5]):.2f}°")
            print(f"笛卡尔: X={float(result[6]):.2f}mm, Y={float(result[7]):.2f}mm, Z={float(result[8]):.2f}mm, "
                  f"Rx={float(result[9]):.2f}°, Ry={float(result[10]):.2f}°, Rz={float(result[11]):.2f}°")
            print("----------------")
        return result

    def _wait_motion_done(self, timeout=30):
        """等待运动完成"""
        start_time = time.time()
        while True:
            result = []
            self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
            if len(result) > 0 and result[0] == 'True':
                print(f"运动完成 (耗时: {time.time() - start_time:.2f}s)")
                return True
            if time.time() - start_time > timeout:
                print("运动超时!")
                return False
            time.sleep(0.1)

    def _safe_move(self, move_func, *args, **kwargs):
        """安全执行运动，包含错误处理和等待"""
        try:
            nRet = move_func(*args, **kwargs)
            print(f"运动指令返回: {nRet}")
            if nRet == 0:
                # 等待运动完成
                self._wait_motion_done()
                # 读取并显示当前位置
                self._read_current_position()
                return True
            else:
                print(f"运动指令失败，错误码: {nRet}")
                # 获取错误码解释
                error_result = []
                self.cps.HRIF_GetErrorCodeStr(self.box_id, nRet, error_result)
                if len(error_result) > 0:
                    print(f"错误描述: {error_result[0]}")
                return False
        except Exception as e:
            print(f"运动异常: {e}")
            return False

    # ============ 关节运动测试 ============

    def test_01_move_j_to_home(self):
        """测试关节运动到零点位置"""
        print("\n" + "=" * 60)
        print("测试1: 关节运动到零点位置 (J1=0, J2=0, J3=15, J4=0, J5=15, J6=0)")

        # 零点位置 (Elfin系列常见零点姿态)
        point = [0, 0,-45, 0, -45, 0]
        raw_acs = [0, 0, -45, 0, -45, 0]
        s_tcp_name = "TCP"
        s_ucs_name = "Base"
        d_velocity = 40
        d_acc = 60
        d_radius = 0
        n_is_use_joint = 1
        n_is_seek = 0
        n_io_bit = 0
        n_io_state = 0
        str_cmd_id = "1"

        success = self._safe_move(
            self.cps.HRIF_MoveJ,
            self.box_id, self.rbt_id,
            point, raw_acs, s_tcp_name, s_ucs_name,
            d_velocity, d_acc, d_radius,
            n_is_use_joint, n_is_seek, n_io_bit, n_io_state, str_cmd_id
        )
        self.assertTrue(success, "移动到零点位置失败")
        print("=" * 60)

    def test_02_move_j_to_custom_pose(self):
        """测试关节运动到自定义姿态"""
        print("\n" + "=" * 60)
        print("测试2: 关节运动到自定义姿态 (J1=30, J2=-20, J3=80, J4=0, J5=70, J6=0)")

        point = [30, -20, 80, 0, 70, 0]
        raw_acs = [30, -20, 80, 0, 70, 0]
        s_tcp_name = "TCP"
        s_ucs_name = "Base"
        d_velocity = 20
        d_acc = 30
        d_radius = 0
        n_is_use_joint = 1
        n_is_seek = 0
        n_io_bit = 0
        n_io_state = 0
        str_cmd_id = "2"

        success = self._safe_move(
            self.cps.HRIF_MoveJ,
            self.box_id, self.rbt_id,
            point, raw_acs, s_tcp_name, s_ucs_name,
            d_velocity, d_acc, d_radius,
            n_is_use_joint, n_is_seek, n_io_bit, n_io_state, str_cmd_id
        )
        self.assertTrue(success, "移动到自定义姿态失败")
        print("=" * 60)

    def test_03_way_point_joint(self):
        """测试路点关节运动 (WayPoint)"""
        print("\n" + "=" * 60)
        print("测试3: 路点关节运动到 (J1=0, J2=0, J3=90, J4=0, J5=90, J6=0)")

        n_move_type = 0  # 0: 关节运动
        point = [0, 0, 90, 0, 90, 0]
        raw_acs = [0, 0, 90, 0, 90, 0]
        s_tcp_name = "TCP"
        s_ucs_name = "Base"
        d_velocity = 30
        d_acc = 30
        d_radius = 0
        n_is_use_joint = 1
        n_is_seek = 0
        n_io_bit = 0
        n_io_state = 0
        str_cmd_id = "3"

        success = self._safe_move(
            self.cps.HRIF_WayPoint,
            self.box_id, self.rbt_id,
            n_move_type, point, raw_acs,
            s_tcp_name, s_ucs_name,
            d_velocity, d_acc, d_radius,
            n_is_use_joint, n_is_seek, n_io_bit, n_io_state, str_cmd_id
        )
        self.assertTrue(success, "路点关节运动失败")
        print("=" * 60)

    # ============ 直线运动测试 ============

    def test_04_move_l_to_position(self):
        """测试直线运动到指定笛卡尔位置"""
        print("\n" + "=" * 60)
        print("测试4: 直线运动到笛卡尔位置 (X=420, Y=0, Z=445, Rx=180, Ry=0, Rz=180)")

        # 读取当前位置作为参考
        current = self._read_current_position()
        if len(current) >= 6:
            raw_acs = [float(current[0]), float(current[1]), float(current[2]),
                       float(current[3]), float(current[4]), float(current[5])]
        else:
            raw_acs = [0, 0, 90, 0, 90, 0]

        point = [420, 0, 445, 180, 0, 180]
        s_tcp_name = "TCP"
        s_ucs_name = "Base"
        d_velocity = 50
        d_acc = 50
        d_radius = 0
        n_is_seek = 0
        n_io_bit = 0
        n_io_state = 0
        str_cmd_id = "4"

        success = self._safe_move(
            self.cps.HRIF_MoveL,
            self.box_id, self.rbt_id,
            point, raw_acs, s_tcp_name, s_ucs_name,
            d_velocity, d_acc, d_radius,
            n_is_seek, n_io_bit, n_io_state, str_cmd_id
        )
        self.assertTrue(success, "直线运动失败")
        print("=" * 60)

    def test_05_way_point_linear(self):
        """测试路点直线运动"""
        print("\n" + "=" * 60)
        print("测试5: 路点直线运动到 (X=400, Y=50, Z=445, Rx=180, Ry=0, Rz=180)")

        n_move_type = 1  # 1: 直线运动
        point = [400, 50, 445, 180, 0, 180]
        raw_acs = [0, 0, 90, 0, 90, 0]
        s_tcp_name = "TCP"
        s_ucs_name = "Base"
        d_velocity = 30
        d_acc = 30
        d_radius = 0
        n_is_use_joint = 0
        n_is_seek = 0
        n_io_bit = 0
        n_io_state = 0
        str_cmd_id = "5"

        success = self._safe_move(
            self.cps.HRIF_WayPoint,
            self.box_id, self.rbt_id,
            n_move_type, point, raw_acs,
            s_tcp_name, s_ucs_name,
            d_velocity, d_acc, d_radius,
            n_is_use_joint, n_is_seek, n_io_bit, n_io_state, str_cmd_id
        )
        self.assertTrue(success, "路点直线运动失败")
        print("=" * 60)

    # ============ 圆弧运动测试 ============

    def test_06_move_c_arc(self):
        """测试圆弧运动"""
        print("\n" + "=" * 60)
        print("测试6: 圆弧运动 (从当前位置经过辅助点到目标位置)")

        # 先移动到起点
        start_point = [420, 0, 445, 180, 0, 180]
        raw_acs = [0, 0, 90, 0, 90, 0]
        s_tcp_name = "TCP"
        s_ucs_name = "Base"

        self.cps.HRIF_MoveL(self.box_id, self.rbt_id,
                            start_point, raw_acs, s_tcp_name, s_ucs_name,
                            50, 50, 0, 0, 0, 0, "start")
        self._wait_motion_done()

        # 圆弧运动
        d_start_point = [420, 0, 445, 180, 0, 180]
        d_aux_point = [420, 50, 445, 180, 0, 180]
        d_end_point = [470, 0, 445, 180, 0, 180]
        n_fixed_posure = 0  # 固定姿态
        n_move_c_type = 1  # 1: 圆弧
        d_rad_len = 0
        d_velocity = 20
        d_acc = 30
        d_radius = 0
        str_cmd_id = "6"

        try:
            nRet = self.cps.HRIF_MoveC(
                self.box_id, self.rbt_id,
                d_start_point, d_aux_point, d_end_point,
                n_fixed_posure, n_move_c_type, d_rad_len,
                d_velocity, d_acc, d_radius,
                s_tcp_name, s_ucs_name, str_cmd_id
            )
            print(f"圆弧运动指令返回: {nRet}")
            if nRet == 0:
                self._wait_motion_done()
                self._read_current_position()
            self.assertEqual(nRet, 0, "圆弧运动失败")
        except Exception as e:
            self.fail(f"圆弧运动异常: {e}")
        print("=" * 60)

    # ============ 相对运动测试 ============

    def test_07_move_rel_j(self):
        """测试关节相对运动"""
        print("\n" + "=" * 60)
        print("测试7: 关节相对运动 (J1正方向移动5°)")

        n_axis = 0  # J1轴
        n_direction = 1  # 正方向
        d_distance = 5.0

        success = self._safe_move(
            self.cps.HRIF_MoveRelJ,
            self.box_id, self.rbt_id,
            n_axis, n_direction, d_distance
        )
        self.assertTrue(success, "关节相对运动失败")
        print("=" * 60)

    def test_08_move_rel_l(self):
        """测试空间相对运动"""
        print("\n" + "=" * 60)
        print("测试8: 空间相对运动 (X正方向移动10mm)")

        n_axis = 0  # X方向
        n_direction = 1  # 正方向
        d_distance = 10.0
        n_tool_motion = 0  # 按用户坐标运动

        try:
            nRet = self.cps.HRIF_MoveRelL(
                self.box_id, self.rbt_id,
                n_axis, n_direction, d_distance, n_tool_motion
            )
            print(f"空间相对运动指令返回: {nRet}")
            if nRet == 0:
                self._wait_motion_done()
                self._read_current_position()
            self.assertEqual(nRet, 0, "空间相对运动失败")
        except Exception as e:
            self.fail(f"空间相对运动异常: {e}")
        print("=" * 60)

    # ============ 多点连续轨迹测试 ============

    def test_09_move_path_j(self):
        """测试连续轨迹运动 (关节空间)"""
        print("\n" + "=" * 60)
        print("测试9: 连续轨迹运动 (关节空间多点轨迹)")

        s_track_name = "TestPath_J"
        d_speed_ratio = 0.3
        d_radius = 5

        # 初始化轨迹
        nRet = self.cps.HRIF_StartPushMovePathJ(
            self.box_id, self.rbt_id,
            s_track_name, d_speed_ratio, d_radius
        )
        print(f"初始化轨迹结果: {nRet}")
        self.assertEqual(nRet, 0, "初始化轨迹失败")

        # 下发轨迹点位 (多个关节位置)
        waypoints = [
            [0, 0, 90, 0, 90, 0],
            [10, -5, 85, 0, 85, 0],
            [20, -10, 80, 0, 80, 0],
            [10, -5, 85, 0, 85, 0],
            [0, 0, 90, 0, 90, 0],
        ]

        for i, wp in enumerate(waypoints):
            nRet = self.cps.HRIF_PushMovePathJ(
                self.box_id, self.rbt_id,
                s_track_name, wp
            )
            print(f"下发点位{i + 1}: {wp}, 结果: {nRet}")
            self.assertEqual(nRet, 0, f"下发点位{i + 1}失败")
            time.sleep(0.1)

        # 结束示教并计算轨迹
        nRet = self.cps.HRIF_EndPushMovePathJ(
            self.box_id, self.rbt_id,
            s_track_name
        )
        print(f"结束示教结果: {nRet}")
        self.assertEqual(nRet, 0, "结束示教失败")

        # 等待轨迹计算完成
        print("等待轨迹计算...")
        time.sleep(2)

        # 执行轨迹
        nRet = self.cps.HRIF_MovePathJ(
            self.box_id, self.rbt_id,
            s_track_name
        )
        print(f"执行轨迹结果: {nRet}")
        self.assertEqual(nRet, 0, "执行轨迹失败")

        # 等待运动完成
        self._wait_motion_done(timeout=60)
        self._read_current_position()

        # 删除轨迹
        nRet = self.cps.HRIF_DelMovePathJ(
            self.box_id, self.rbt_id,
            s_track_name
        )
        print(f"删除轨迹结果: {nRet}")
        print("=" * 60)

    # ============ WayPoint2 测试 (支持圆弧过渡) ============

    def test_10_way_point2_linear(self):
        """测试WayPoint2直线运动 (支持过渡不减速)"""
        print("\n" + "=" * 60)
        print("测试10: WayPoint2直线运动到目标位置")

        n_move_type = 1  # 1: 直线运动
        end_pos = [400, 0, 445, 180, 0, 180]
        aux_pos = [0, 0, 0, 0, 0, 0]  # 直线运动时无效
        acs_pose = [0, 0, 90, 0, 90, 0]
        s_tcp_name = "TCP"
        s_ucs_name = "Base"
        d_velocity = 30
        d_acc = 40
        d_radius = 10  # 过渡半径
        n_is_use_joint = 0
        n_is_seek = 0
        n_io_bit = 0
        n_io_state = 0
        str_cmd_id = "10"

        try:
            nRet = self.cps.HRIF_WayPoint2(
                self.box_id, self.rbt_id,
                n_move_type, end_pos, aux_pos, acs_pose,
                s_tcp_name, s_ucs_name,
                d_velocity, d_acc, d_radius,
                n_is_use_joint, n_is_seek, n_io_bit, n_io_state, str_cmd_id
            )
            print(f"WayPoint2直线运动返回: {nRet}")
            if nRet == 0:
                self._wait_motion_done()
                self._read_current_position()
            self.assertEqual(nRet, 0, "WayPoint2直线运动失败")
        except Exception as e:
            self.fail(f"WayPoint2直线运动异常: {e}")
        print("=" * 60)

    # ============ 移动到安全位置 ============

    def test_11_move_to_safe_position(self):
        """测试移动到安全位置"""
        print("\n" + "=" * 60)
        print("测试11: 移动到安全位置 (MoveToSS)")

        try:
            nRet = self.cps.HRIF_MoveToSS(self.box_id)
            print(f"移动到安全位置结果: {nRet}")
            if nRet == 0:
                self._wait_motion_done()
                self._read_current_position()
            self.assertEqual(nRet, 0, "移动到安全位置失败")
        except Exception as e:
            self.fail(f"移动到安全位置异常: {e}")
        print("=" * 60)

    # ============ 短点动测试 ============

    def test_12_short_jog_j(self):
        """测试关节短点动"""
        print("\n" + "=" * 60)
        print("测试12: 关节短点动 (J1正方向)")

        n_axis_id = 0
        n_direction = 1

        try:
            nRet = self.cps.HRIF_ShortJogJ(
                self.box_id, self.rbt_id,
                n_axis_id, n_direction
            )
            print(f"关节短点动返回: {nRet}")
            if nRet == 0:
                self._wait_motion_done()
                self._read_current_position()
            self.assertEqual(nRet, 0, "关节短点动失败")
        except Exception as e:
            self.fail(f"关节短点动异常: {e}")
        print("=" * 60)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)
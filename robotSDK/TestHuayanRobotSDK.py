import unittest
import time
from CPS import CPSClient


class TestHuayanRobotSDK(unittest.TestCase):
    """华沿机器人SDK单元测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化 - 连接机器人"""
        cls.cps = CPSClient()
        cls.box_id = 0
        cls.rbt_id = 0
        cls.ip = '192.168.0.10'
        cls.port = 10003

        # 尝试连接
        try:
            cls.nRet = cls.cps.HRIF_Connect(cls.box_id, cls.ip, cls.port)
            print(f"连接结果: {cls.nRet}")
        except Exception as e:
            print(f"连接异常: {e}")
            cls.nRet = -1

    @classmethod
    def tearDownClass(cls):
        """测试类清理 - 断开连接"""
        try:
            cls.cps.HRIF_DisConnect(cls.box_id)
            print("已断开连接")
        except Exception as e:
            print(f"断开连接异常: {e}")

    def test_01_connect(self):
        """测试连接功能"""
        result = self.cps.HRIF_IsConnected(self.box_id)
        print(f"连接状态: {result}")
        self.assertIsNotNone(result)

    def test_02_read_version(self):
        """测试读取版本信息"""
        result = []
        nRet = self.cps.HRIF_ReadVersion(self.box_id, self.rbt_id, result)
        print(f"读取版本结果: {nRet}")
        print(f"版本信息: {result}")
        self.assertEqual(nRet, 0)
        self.assertGreater(len(result), 0)

    def test_03_read_robot_model(self):
        """测试读取机器人型号"""
        result = []
        nRet = self.cps.HRIF_ReadRobotModel(self.box_id, self.rbt_id, result)
        print(f"读取型号结果: {nRet}")
        print(f"机器人型号: {result}")
        self.assertEqual(nRet, 0)

    def test_04_read_robot_state(self):
        """测试读取机器人状态"""
        result = []
        nRet = self.cps.HRIF_ReadRobotState(self.box_id, self.rbt_id, result)
        print(f"读取状态结果: {nRet}")
        print(f"机器人状态: {result}")
        self.assertEqual(nRet, 0)

    def test_05_read_act_pos(self):
        """测试读取实际位置"""
        result = []
        nRet = self.cps.HRIF_ReadActPos(self.box_id, self.rbt_id, result)
        print(f"读取位置结果: {nRet}")
        if nRet == 0 and len(result) >= 6:
            print(
                f"关节位置: J1={result[0]}, J2={result[1]}, J3={result[2]}, J4={result[3]}, J5={result[4]}, J6={result[5]}")
            print(
                f"笛卡尔位置: X={result[6]}, Y={result[7]}, Z={result[8]}, Rx={result[9]}, Ry={result[10]}, Rz={result[11]}")
        self.assertEqual(nRet, 0)

    def test_06_read_cur_tcp(self):
        """测试读取当前工具坐标"""
        result = []
        nRet = self.cps.HRIF_ReadCurTCP(self.box_id, self.rbt_id, result)
        print(f"读取工具坐标结果: {nRet}")
        if nRet == 0 and len(result) >= 6:
            print(
                f"工具坐标: X={result[0]}, Y={result[1]}, Z={result[2]}, Rx={result[3]}, Ry={result[4]}, Rz={result[5]}")
        self.assertEqual(nRet, 0)

    def test_07_read_cur_ucs(self):
        """测试读取当前用户坐标"""
        result = []
        nRet = self.cps.HRIF_ReadCurUCS(self.box_id, self.rbt_id, result)
        print(f"读取用户坐标结果: {nRet}")
        if nRet == 0 and len(result) >= 6:
            print(
                f"用户坐标: X={result[0]}, Y={result[1]}, Z={result[2]}, Rx={result[3]}, Ry={result[4]}, Rz={result[5]}")
        self.assertEqual(nRet, 0)

    def test_08_read_override(self):
        """测试读取速度比"""
        result = []
        nRet = self.cps.HRIF_ReadOverride(self.box_id, self.rbt_id, result)
        print(f"读取速度比结果: {nRet}")
        if nRet == 0 and len(result) > 0:
            print(f"当前速度比: {result[0]}")
        self.assertEqual(nRet, 0)

    def test_09_read_payload(self):
        """测试读取负载参数"""
        result = []
        nRet = self.cps.HRIF_ReadPayload(self.box_id, result)
        print(f"读取负载结果: {nRet}")
        if nRet == 0 and len(result) >= 4:
            print(f"负载参数: 质量={result[0]}kg, X={result[1]}mm, Y={result[2]}mm, Z={result[3]}mm")
        self.assertEqual(nRet, 0)

    def test_10_read_max_payload(self):
        """测试读取最大负载"""
        result = []
        nRet = self.cps.HRIF_ReadMaxPayload(self.box_id, result)
        print(f"读取最大负载结果: {nRet}")
        if nRet == 0 and len(result) > 0:
            print(f"最大负载: {result[0]}kg")
        self.assertEqual(nRet, 0)

    def test_11_read_emergency_info(self):
        """测试读取急停信息"""
        result = []
        nRet = self.cps.HRIF_ReadEmergencyInfo(self.box_id, self.rbt_id, result)
        print(f"读取急停信息结果: {nRet}")
        if nRet == 0 and len(result) >= 4:
            print(f"急停信息: 急停错误={result[0]}, 急停信号={result[1]}, 光幕错误={result[2]}, 光幕信号={result[3]}")
        self.assertEqual(nRet, 0)

    def test_12_read_joint_max_vel(self):
        """测试读取关节最大速度"""
        result = []
        nRet = self.cps.HRIF_ReadJointMaxVel(self.box_id, self.rbt_id, result)
        print(f"读取关节最大速度结果: {nRet}")
        if nRet == 0 and len(result) >= 6:
            print(
                f"关节最大速度: J1={result[0]}, J2={result[1]}, J3={result[2]}, J4={result[3]}, J5={result[4]}, J6={result[5]}")
        self.assertEqual(nRet, 0)

    def test_13_read_linear_max_speed(self):
        """测试读取直线最大速度"""
        result = []
        nRet = self.cps.HRIF_ReadLinearMaxSpeed(self.box_id, self.rbt_id, result)
        print(f"读取直线最大速度结果: {nRet}")
        if nRet == 0 and len(result) >= 3:
            print(f"直线参数: 速度={result[0]}mm/s, 加速度={result[1]}mm/s², 加加速度={result[2]}mm/s³")
        self.assertEqual(nRet, 0)

    def test_14_read_force_control_state(self):
        """测试读取力控状态"""
        result = []
        nRet = self.cps.HRIF_ReadForceControlState(self.box_id, self.rbt_id, result)
        print(f"读取力控状态结果: {nRet}")
        if nRet == 0 and len(result) > 0:
            states = {0: "关闭状态", 1: "开力控探寻状态", 2: "力控探寻完成状态", 3: "力控自由驱动状态"}
            print(f"力控状态: {states.get(int(result[0]), '未知')}")
        self.assertEqual(nRet, 0)

    def test_15_read_box_info(self):
        """测试读取电箱信息"""
        result = []
        nRet = self.cps.HRIF_ReadBoxInfo(self.box_id, result)
        print(f"读取电箱信息结果: {nRet}")
        if nRet == 0 and len(result) >= 6:
            print(f"电箱信息: 连接={result[0]}, 48V电压状态={result[1]}, 输出电压={result[2]}V, 输出电流={result[3]}A")
        self.assertEqual(nRet, 0)

    def test_16_set_override(self):
        """测试设置速度比"""
        # 设置速度比为0.5
        dOverride = 0.03
        nRet = self.cps.HRIF_SetOverride(self.box_id, self.rbt_id, dOverride)
        print(f"设置速度比结果: {nRet}")
        self.assertEqual(nRet, 0)

        # 验证设置
        result = []
        self.cps.HRIF_ReadOverride(self.box_id, self.rbt_id, result)
        if len(result) > 0:
            print(f"验证速度比: {result[0]}")

    def test_17_set_tool_motion(self):
        """测试设置Tool坐标系运动模式"""
        # 开启Tool坐标系运动模式
        nState = 1
        nRet = self.cps.HRIF_SetToolMotion(self.box_id, self.rbt_id, nState)
        print(f"设置Tool运动模式结果: {nRet}")
        self.assertEqual(nRet, 0)

        # 恢复关闭
        time.sleep(0.5)
        nRet = self.cps.HRIF_SetToolMotion(self.box_id, self.rbt_id, 0)
        print(f"恢复Tool运动模式结果: {nRet}")

    def test_18_quaternion2rpy(self):
        """测试四元素转欧拉角"""
        result = []
        dQuaW, dQuaX, dQuaY, dQuaZ = 1.0, 0.0, 0.0, 0.0
        nRet = self.cps.HRIF_Quaternion2RPY(self.box_id, dQuaW, dQuaX, dQuaY, dQuaZ, result)
        print(f"四元素转欧拉角结果: {nRet}")
        if nRet == 0 and len(result) >= 3:
            print(f"欧拉角: Rx={result[0]}°, Ry={result[1]}°, Rz={result[2]}°")
        self.assertEqual(nRet, 0)

    def test_19_rpy2quaternion(self):
        """测试欧拉角转四元素"""
        result = []
        dRx, dRy, dRz = 0.0, 0.0, 0.0
        nRet = self.cps.HRIF_RPY2Quaternion(self.box_id, dRx, dRy, dRz, result)
        print(f"欧拉角转四元素结果: {nRet}")
        if nRet == 0 and len(result) >= 4:
            print(f"四元素: W={result[0]}, X={result[1]}, Y={result[2]}, Z={result[3]}")
        self.assertEqual(nRet, 0)

    def test_20_is_connected(self):
        """测试连接状态判断"""
        result = self.cps.HRIF_IsConnected(self.box_id)
        print(f"连接状态: {result}")
        self.assertIsNotNone(result)


class TestHuayanRobotMotion(unittest.TestCase):
    """运动控制相关测试（需要机器人使能状态）"""

    @classmethod
    def setUpClass(cls):
        cls.cps = CPSClient()
        cls.box_id = 0
        cls.rbt_id = 0
        cls.ip = '192.168.0.10'
        cls.port = 10003

        try:
            cls.cps.HRIF_Connect(cls.box_id, cls.ip, cls.port)
            # 尝试使能
            cls.cps.HRIF_GrpEnable(cls.box_id, cls.rbt_id)
            time.sleep(1)
        except Exception as e:
            print(f"初始化异常: {e}")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.cps.HRIF_GrpDisable(cls.box_id, cls.rbt_id)
            cls.cps.HRIF_DisConnect(cls.box_id)
        except Exception:
            pass

    def test_01_is_motion_done(self):
        """测试运动完成状态"""
        result = []
        nRet = self.cps.HRIF_IsMotionDone(self.box_id, self.rbt_id, result)
        print(f"运动完成状态: {nRet}, 结果: {result}")
        self.assertEqual(nRet, 0)

    def test_02_is_blending_done(self):
        """测试路点运动完成状态"""
        result = []
        nRet = self.cps.HRIF_IsBlendingDone(self.box_id, self.rbt_id, result)
        print(f"路点运动完成状态: {nRet}, 结果: {result}")
        self.assertEqual(nRet, 0)

    def test_03_move_rel_j(self):
        """测试关节相对运动"""
        # J1轴正方向移动1度
        nAxis = 0
        nDirection = 1
        dDistance = 1.0

        nRet = self.cps.HRIF_MoveRelJ(self.box_id, self.rbt_id, nAxis, nDirection, dDistance)
        print(f"关节相对运动结果: {nRet}")
        self.assertEqual(nRet, 0)

        # 等待运动完成
        time.sleep(2)

    def test_04_move_to_ss(self):
        """测试移动到安全位置"""
        nRet = self.cps.HRIF_MoveToSS(self.box_id)
        print(f"移动到安全位置结果: {nRet}")
        self.assertEqual(nRet, 0)

    def test_05_grp_stop(self):
        """测试停止运动"""
        nRet = self.cps.HRIF_GrpStop(self.box_id, self.rbt_id)
        print(f"停止运动结果: {nRet}")
        self.assertEqual(nRet, 0)


class TestHuayanRobotIO(unittest.TestCase):
    """IO控制相关测试"""

    @classmethod
    def setUpClass(cls):
        cls.cps = CPSClient()
        cls.box_id = 0
        cls.rbt_id = 0
        cls.ip = '192.168.0.10'
        cls.port = 10003

        try:
            cls.cps.HRIF_Connect(cls.box_id, cls.ip, cls.port)
        except Exception as e:
            print(f"初始化异常: {e}")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.cps.HRIF_DisConnect(cls.box_id)
        except Exception:
            pass

    def test_01_read_box_di(self):
        """测试读取电箱数字输入"""
        result = []
        nBit = 0
        nRet = self.cps.HRIF_ReadBoxDI(self.box_id, nBit, result)
        print(f"读取DI{0}结果: {nRet}, 状态: {result}")
        self.assertEqual(nRet, 0)

    def test_02_read_box_do(self):
        """测试读取电箱数字输出"""
        result = []
        nBit = 0
        nRet = self.cps.HRIF_ReadBoxDO(self.box_id, nBit, result)
        print(f"读取DO{0}结果: {nRet}, 状态: {result}")
        self.assertEqual(nRet, 0)

    def test_03_set_box_do(self):
        """测试设置电箱数字输出"""
        # 设置DO0为高电平
        nBit = 0
        nVal = 1
        nRet = self.cps.HRIF_SetBoxDO(self.box_id, nBit, nVal)
        print(f"设置DO{0}结果: {nRet}")
        self.assertEqual(nRet, 0)

        # 验证
        time.sleep(0.5)
        result = []
        self.cps.HRIF_ReadBoxDO(self.box_id, nBit, result)
        print(f"验证DO{0}状态: {result}")

        # 恢复低电平
        self.cps.HRIF_SetBoxDO(self.box_id, nBit, 0)

    def test_04_read_end_di(self):
        """测试读取末端数字输入"""
        result = []
        nBit = 0
        nRet = self.cps.HRIF_ReadEndDI(self.box_id, self.rbt_id, nBit, result)
        print(f"读取末端DI{0}结果: {nRet}, 状态: {result}")
        self.assertEqual(nRet, 0)

    def test_05_read_end_do(self):
        """测试读取末端数字输出"""
        result = []
        nBit = 0
        nRet = self.cps.HRIF_ReadEndDO(self.box_id, self.rbt_id, nBit, result)
        print(f"读取末端DO{0}结果: {nRet}, 状态: {result}")
        self.assertEqual(nRet, 0)

    def test_06_set_end_do(self):
        """测试设置末端数字输出"""
        nBit = 0
        nVal = 1
        nRet = self.cps.HRIF_SetEndDO(self.box_id, self.rbt_id, nBit, nVal)
        print(f"设置末端DO{0}结果: {nRet}")
        self.assertEqual(nRet, 0)

        # 恢复
        time.sleep(0.5)
        self.cps.HRIF_SetEndDO(self.box_id, self.rbt_id, nBit, 0)


if __name__ == '__main__':
    # 运行所有测试
    # unittest.main()

    # 运行特定测试
    # unittest.main(defaultTest='TestHuayanRobotSDK.test_01_connect')

    # 运行所有测试并显示详细信息
    unittest.main(verbosity=2)
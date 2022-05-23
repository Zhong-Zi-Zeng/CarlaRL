from CarlaApiAsync import CarlaApi
from DQN import Agent
from GUI import GUI
from SegNetwork.EncodeAndFlatten import Network
import matplotlib.pyplot as plt
import carla
import cv2
import numpy as np
import os
import time
import pygame

# 動作對照表
action_chart = {
    0: 'Forward',
    1: 'Stop',
    2: 'Half left turn',
    3: 'Half right turn',
    4: 'Full left turn',
    5: 'Full right turn',
}

class main:
    def __init__(self):
        self.CarlaApi = CarlaApi(img_width=400,
                                 img_height=300,
                                 MIN_MIDDLE_DIS=0.75)
        self.DQN = Agent(alpha=0.0005,
                         gamma=0.99,
                         n_actions=6,
                         epsilon=0.4,
                         batch_size=16,
                         epsilon_end=0.1,
                         mem_size=100000,
                         epsilon_dec=0.96,
                         input_shape=24,
                         iteration=100,
                         use_pri=True)
        # 載入上次權重並繼續訓練
        # self.DQN.load_model()
        self.GUI = GUI()
        # 期望時速
        self.DESIRED_SPEED = 8
        # 與道路中心點最遠允許距離
        self.MAX_MIDDLE_DIS = 5
        # 允許偏移角度
        self.DEGREE_LIMIT = 130
        # 全連接網路輸出閥值
        self.THRESHOLD = 0.8
        # 總訓練次數
        self.EPISODES = 100000
        # 預測變數
        self.pre_tl = None
        self.pre_need_slow = None
        self.pre_tl_dis = None
        # 載入編碼網路及全連接層網路
        self.now_path = os.getcwd().replace('\\','/') + '/SegNetwork'
        self.EncodeAndFlattenNetwork = Network(now_path=self.now_path).buildModel()
        self.EncodeAndFlattenNetwork.load_weights()

        self.train()

    def get_image(self):
        camera_data = self.CarlaApi.camera_data()
        front_bgr_frame = camera_data['front_bgr_camera']
        top_bgr_frame = camera_data['top_bgr_camera']
        seg_frame = camera_data['seg_camera']

        return front_bgr_frame, top_bgr_frame, seg_frame

    def get_state(self,bgr_frame):
        car_data = self.CarlaApi.car_data()

        # 交通狀況
        self.pre_need_slow, self.pre_tl, self.pre_tl_dis = self.EncodeAndFlattenNetwork.predict(bgr_frame)
        self.pre_need_slow = np.squeeze(self.pre_need_slow)
        self.pre_tl = np.squeeze(self.pre_tl)
        self.pre_tl_dis = np.squeeze(self.pre_tl_dis)

        NeedSlow = 1 if self.pre_need_slow > self.THRESHOLD else 0
        TL = np.zeros(3)
        TL[np.argmax(self.pre_tl)] = 1
        TL_dis = np.zeros(4)
        TL_dis[np.argmax(self.pre_tl_dis)] = 1

        # 角度部分
        car_data['way_degree'] = np.clip(car_data['way_degree'], -60, 60)
        degree_lin = np.linspace(-60, 60, 10)
        degree_state = np.zeros(10)
        for i in range(len(degree_lin)):
            if car_data['way_degree'] < degree_lin[i]:
                degree_state[i - 1] = 1
                break

        # 距離部分
        car_data['way_dis'] = np.clip(car_data['way_dis'], 0.6, 5)
        dis_lin = np.linspace(0.6,5,5)
        dis_state = np.zeros(5)
        for i in range(len(dis_lin)):
            if car_data['way_dis'] < dis_lin[i]:
                dis_state[i - 1] = 1
                break

        state = np.hstack((degree_state,dis_state,car_data['car_speed'],NeedSlow,TL,TL_dis))
        return state

    def show_state(self):
        car_data = self.CarlaApi.car_data()
        del car_data['tl']

        car_data['car_speed'] = str(round(car_data['car_speed'],2)) + ' km/h'
        car_data['way_degree'] = str(round(car_data['way_degree'],2)) + ' °'
        car_data['way_dis'] = str(round(car_data['way_dis'],2)) + ' m'

        # 交通狀況
        Pre_needslow = 'True' if self.pre_need_slow > self.THRESHOLD else 'False'

        if np.argmax(self.pre_tl) == 0:
            Pre_TL = 'Green'
        elif np.argmax(self.pre_tl) == 1:
            Pre_TL = 'Red'
        else:
            Pre_TL = 'None'

        if np.argmax(self.pre_tl_dis) == 0:
            Pre_TL_dis = 'Close'
        elif np.argmax(self.pre_tl_dis) == 1:
            Pre_TL_dis = 'Medium'
        elif np.argmax(self.pre_tl_dis) == 2:
            Pre_TL_dis = 'Far'
        else:
            Pre_TL_dis = 'None'

        car_data['Pre_needslow'] = Pre_needslow
        car_data['Pre_TL'] = Pre_TL
        car_data['Pre_TL_Dis'] = Pre_TL_dis

        return car_data

    def train(self):
        self.CarlaApi.initial()
        self.CarlaApi.wait_for_sim()
        total_reward_list = []

        try:
            for i in range(self.EPISODES):
                done = False
                total_reward = 0
                # St時刻的狀態
                front_bgr_frame, top_bgr_frame, seg_frame = self.get_image()
                state = self.get_state(front_bgr_frame)

                while not done:
                    # 選取動作
                    action = self.DQN.choose_action(state)
                    self.control_car(action)

                    # 顯示影像
                    self.GUI.clear()
                    self.GUI.draw_image(top_bgr_frame)
                    self.GUI.draw_text_info(self.show_state(),
                                            action=action_chart[action],
                                            episode=i)
                    if self.GUI.should_quit():
                        return

                    # 計算獎勵
                    reward, done = self.compute_reward()
                    total_reward += reward

                    # St+1時刻的影像
                    next_front_bgr_frame, next_top_bgr_frame, next_seg_frame = self.get_image()
                    next_state = self.get_state(next_front_bgr_frame)

                    # 訓練網路
                    self.DQN.remember(state, action, reward, next_state, done)
                    self.DQN.learn()

                    # 更改狀態
                    state = next_state
                    top_bgr_frame = next_top_bgr_frame

                    pygame.display.update()

                total_reward_list.append(total_reward)
                if i % 10 == 0:
                    self.DQN.save_model()

                self.CarlaApi.reset()
                time.sleep(0.5)
        finally:
            cv2.destroyAllWindows()
            plt.plot(total_reward_list)
            plt.show()
            self.CarlaApi.destroy()
            print('Destroy actor')

    """計算獎勵"""
    def compute_reward(self):
        sensor_data = self.CarlaApi.sensor_data()
        car_data = self.CarlaApi.car_data()
        done = False
        reward = 0

        # 速度達標準
        if car_data['tl']:
            if int(car_data['car_speed']) == 0:
                reward += 1
        # 速度未達標準
        elif int(car_data['car_speed']) == 0:
            reward += -1.5

        # 判斷位置獎勵
        reward += np.exp(-abs(car_data['way_degree']) / 15) * 1.5

        # 中止訓練
        if car_data['way_dis'] > self.MAX_MIDDLE_DIS or \
            abs(car_data['way_degree']) > self.DEGREE_LIMIT or \
            sensor_data['collision_sensor'] or \
            car_data['tl'] and int(car_data['car_speed']) != 0 :
            reward = -10
            done = True

        return reward, done


    """車輛控制"""
    def control_car(self,action):
        """:param
                0:前進、1:煞車、2:半左轉、3:半右轉、4:全左轉、5:全右轉
        """
        control = carla.VehicleControl()
        control.throttle = 0.35
        control.brake = 0
        if (action == 0):
            control.steer = 0.0
        elif (action == 1):
            control.throttle = 0.0
            control.brake = 1.0
        elif (action == 2):
            control.steer = -0.3
        elif (action == 3):
            control.steer = 0.3
        elif (action == 4):
            control.steer = -0.7
        elif (action == 5):
            control.steer = 0.7

        self.CarlaApi.control_vehicle(control)

if __name__ == "__main__":
    main()


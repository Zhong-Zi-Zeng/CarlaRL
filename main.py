from CarlaApiAsync import CarlaApi
from DQN import Agent
import matplotlib.pyplot as plt
import carla
import cv2
import numpy as np
import copy
import time


def process_seg_frame(seg_frame):
    seg_frame_cp = copy.deepcopy(seg_frame)
    AllClass = {
        '0': (0, 0, 0),  # 未標記
        '1': (70, 70, 70),  # 建築
        '2': (40, 40, 100),  # 柵欄
        '3': (80, 90, 55),  # 其他
        '4': (60, 20, 220),  # 行人
        '5': (153, 153, 153),  # 桿
        '6': (50, 234, 157),  # 道路線
        '7': (128, 64, 128),  # 馬路
        '8': (232, 35, 244),  # 人行道
        '9': (35, 142, 107),  # 植披
        '10': (142, 0, 0),  # 汽車
        '11': (156, 102, 102),  # 牆
        '12': (0, 220, 220),  # 交通號誌
        '13': (180, 130, 70),  # 天空
        '14': (81, 0, 81),  # 地面
        '15': (100, 100, 150),  # 橋
        '16': (140, 150, 230),  # 鐵路
        '17': (180, 165, 180),  # 護欄
        '18': (30, 170, 250),  # 紅綠燈
        '19': (160, 190, 110),  # 靜止的物理
        '20': (50, 120, 170),  # 動態的
        '21': (150, 60, 45),  # 水
        '22': (100, 170, 145)  # 地形
    }
    # not_important = [2, 3, 5, 12, 13, 15, 16, 17, 18, 19, 21, 22]
    not_important = [1, 2, 3, 5, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    seg_frame_height = seg_frame.shape[0]
    seg_frame_width = seg_frame.shape[1]
    for item in AllClass.items():
        i = int(item[0])
        if i in not_important:
            b,g,r = item[1]
            matrix = np.where((seg_frame[:, :, 0] == b) & (seg_frame[:, :, 1] == g) & (seg_frame[:, :, 2] == r),
                                   np.ones((seg_frame_height, seg_frame_width)), np.zeros((seg_frame_height, seg_frame_width)))
            seg_frame_cp[matrix == 1] = 0

    return seg_frame_cp

class main:
    def __init__(self):
        self.CarlaApi = CarlaApi(img_width=100,img_height=100)
        self.DQN = Agent(lr=0.0003,
                         gamma=0.99,
                         n_actions=6,
                         epsilon=0.3,
                         batch_size=8,
                         epsilon_end=0.1,
                         mem_size=3000,
                         epsilon_dec=0.95,
                         img_width=100,
                         img_height=100,
                         iteration=200,
                         fixed_q=True)
        # 期望時速
        self.DESIRED_SPEED = 15
        # 與道路中心點最遠允許距離
        self.MAX_MIDDLE_DIS = 2

        self.EPISODES = 10000
        self.train()

    def get_image(self):
        camera_data = self.CarlaApi.camera_data()
        bgr_frame = camera_data['bgr_camera']
        seg_frame = camera_data['seg_camera']

        return bgr_frame, seg_frame

    def train(self):
        self.CarlaApi.initial()
        self.CarlaApi.wait_for_sim()
        total_reward_list = []
        old_total_reward = 0
        try:
            for i in range(self.EPISODES):
                done = False
                print('Episode:%d'%(i))
                total_reward = 0
                while not done:
                    # St時刻的影像
                    bgr_frame, seg_frame = self.get_image()
                    seg_frame = process_seg_frame(seg_frame)

                    # 顯示影像
                    show_frame = np.hstack((seg_frame, bgr_frame))
                    cv2.imshow("", show_frame)
                    if cv2.waitKey(1) == ord('q'):
                        exit()

                    # 選取動作
                    action = self.DQN.choose_action(seg_frame / 255)
                    self.control_car(action)

                    # 計算獎勵
                    reward,done = self.compute_reward()
                    total_reward += reward

                    # St+1時刻的影像
                    next_bgr_frame, next_seg_frame = self.get_image()
                    next_seg_frame = process_seg_frame(next_seg_frame)

                    # 訓練網路
                    self.DQN.remember(seg_frame/255, action, reward, next_seg_frame/255, done)
                    self.DQN.learn()

                total_reward_list.append(total_reward)
                if total_reward > old_total_reward:
                    old_total_reward = total_reward
                    self.DQN.save_model()

                self.CarlaApi.reset()
                time.sleep(0.5)
        finally:
            self.CarlaApi.destroy()
            cv2.destroyAllWindows()
            plt.plot(total_reward_list)
            plt.show()
            print('Destroy actor')

    """計算獎勵"""
    def compute_reward(self):
        sensor_data = self.CarlaApi.sensor_data()
        # lane_line_info = sensor_data['lane_line_sensor']
        collision_info = sensor_data['collision_sensor']
        # traffic_info = sensor_data['traffic_info']
        dis_info = sensor_data['finish_point_dis']
        car_speed = sensor_data['car_speed']
        done = False

        # 速度獎勵
        if car_speed == self.DESIRED_SPEED:
            speed_reward = 1
        else:
            speed_reward = 0

        # 與目標的距離獎勵
        if dis_info <= 0.3:
            dis_reward = 0
        else:
            dis_reward = -1

        reward = dis_reward + speed_reward

        # 懲罰
        if collision_info or dis_info > self.MAX_MIDDLE_DIS:
            done = True
            reward = -1

        print('=' * 30)
        print('Target Distance:%.3f' % (dis_info))
        print('Car Speed:%.2f' % (car_speed))
        print('Reward:%.3f' % (reward))

        return reward, done


    """車輛控制"""
    def control_car(self,action):
        """:param
                0:前進、1:煞車、2:半左轉、3:半右轉、4:全左轉、5:全右轉
        """
        control = carla.VehicleControl()
        control.throttle = 0.7
        control.brake = 0
        if (action == 0):
            control.steer = 0.0
        elif (action == 1):
            control.throttle = 0.0
            control.brake = 1.0
        elif (action == 2):
            control.steer = -0.5
        elif (action == 3):
            control.steer = 0.5
        elif (action == 4):
            control.steer = -1.0
        elif (action == 5):
            control.steer = 1.0


        self.CarlaApi.control_vehicle(control)


if __name__ == "__main__":
    main()


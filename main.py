from CarlaApiAsync import CarlaApi
from DQN import Agent
from SegNetwork.EncodeAndFlatten import Network
import matplotlib.pyplot as plt
import carla
import cv2
import numpy as np
import os


class main:
    def __init__(self):
        self.CarlaApi = CarlaApi(img_width=400,img_height=300)
        self.DQN = Agent(lr=0.0003,
                         gamma=0.99,
                         n_actions=6,
                         epsilon=0.3,
                         batch_size=16,
                         epsilon_end=0.1,
                         mem_size=40000,
                         epsilon_dec=0.95,
                         input_shape=6)

        # 期望時速
        self.DESIRED_SPEED = 20
        # 與道路中心點最遠允許距離
        self.MAX_MIDDLE_DIS = 3.5
        # 與道路中心點完成距離
        self.MIN_MIDDLE_DIS = 0.6
        # 允許偏移角度
        self.DEGREE_LIMIT = 15
        # 編碼器輸出閥值
        self.THRESHOLD = 0.8

        self.EPISODES = 100000
        self.now_path = os.getcwd().replace('\\','/') + '/SegNetwork'
        self.EncodeAndFlattenNetwork = Network(now_path=self.now_path).buildModel()
        self.EncodeAndFlattenNetwork.load_weights()

        self.train()

    def get_image(self):
        camera_data = self.CarlaApi.camera_data()
        bgr_frame = camera_data['bgr_camera']
        seg_frame = camera_data['seg_camera']

        return bgr_frame, seg_frame

    def get_state(self,bgr_frame):
        tl, junction = self.EncodeAndFlattenNetwork.predict(bgr_frame)
        tl = np.squeeze(tl)
        junction = np.squeeze(junction)
        tl = 1 if tl > self.THRESHOLD else 0
        junction = 1 if junction > self.THRESHOLD else 0

        car_data = self.CarlaApi.car_data()
        state = [tl, junction, car_data['car_speed'], car_data['car_steering'], car_data['way_dis'],
                 car_data['way_degree']/360]
        print(car_data)
        return state

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
                    # St時刻的狀態
                    bgr_frame, _ = self.get_image()
                    state = self.get_state(bgr_frame)

                    # 顯示影像
                    cv2.imshow("", bgr_frame)
                    if cv2.waitKey(1) == ord('q'):
                        exit()

                    # 選取動作
                    action = self.DQN.choose_action(state)
                    self.control_car(action)

                    # 計算獎勵
                    reward, done = self.compute_reward()
                    total_reward += reward

                    # St+1時刻的影像
                    next_bgr_frame, _ = self.get_image()
                    next_state = self.get_state(next_bgr_frame)

                    # 訓練網路
                    self.DQN.remember(state, action, reward, next_state, done)
                    self.DQN.learn()

                total_reward_list.append(total_reward)

                if total_reward > old_total_reward:
                    old_total_reward = total_reward
                    self.DQN.save_model()

                self.CarlaApi.reset()
        finally:
            self.CarlaApi.destroy()
            cv2.destroyAllWindows()
            plt.plot(total_reward_list)
            plt.show()
            print('Destroy actor')

    """計算獎勵"""
    def compute_reward(self):
        sensor_data = self.CarlaApi.sensor_data()
        car_data = self.CarlaApi.car_data()
        collision = sensor_data['collision_sensor']
        done = False
        reward = 0

        # 紅燈時改變速度期望值
        if str(car_data['tl']) == 'Red':
            if car_data['car_speed'] == 0:
                reward += 1
        elif str(car_data['tl']) == 'Green':
            if int(car_data['car_speed']) == self.DESIRED_SPEED:
                reward += 1
        elif str(car_data['tl']) == 'Green' and int(car_data['car_speed']) == 0:
            reward -= 1

        # 判斷位置獎勵
        if car_data['way_dis'] > self.MAX_MIDDLE_DIS:
            reward = -1
            done = True

        # 若到達指定距離則切換下一個中心點
        if car_data['way_dis'] < self.MIN_MIDDLE_DIS:
            reward += 1
            self.CarlaApi.toggle_waypoint()

        # 判斷角度
        if abs(car_data['way_degree']) > self.DEGREE_LIMIT:
            reward = -1
            done = True

        # 是否有撞擊到東西
        if collision:
            reward = -1
            done = True

        return reward, done


    """車輛控制"""
    def control_car(self,action):
        """:param
                0:前進、1:煞車、2:半左轉、3:半右轉、4:全左轉、5:全右轉
        """
        control = carla.VehicleControl()
        control.throttle = 0.6
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


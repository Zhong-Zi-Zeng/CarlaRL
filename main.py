from CarlaApi import CarlaApi
from Actor_Critic import Actor_Critic
import carla
import cv2
import numpy as np


class main:
    def __init__(self):
        self.CarlaApi = CarlaApi(host='localhost',
                                 image_width=100,
                                 image_height=100,
                                 queue_max_size=5,
                                 fps=30)

        self.ActorCritic = Actor_Critic(n_actions=6)
        self.MIN_SPEED = 1
        self.MAX_SPEED = 30
        self.EPISODES = 10000

        self.train()

    def get_image(self):
        sensor_data = self.CarlaApi.camera_data(2.0)
        rgb_frame = sensor_data[0]
        seg_frame = sensor_data[1]

        return rgb_frame, seg_frame

    def train(self):
        self.CarlaApi.initial()
        try:
            for i in range(self.EPISODES):
                done = False
                print('Episode:%d'%(i))
                
                while not done:
                    self.CarlaApi.tick()
                    rgb_frame, seg_frame = self.get_image()
    
                    action = self.ActorCritic.choose_action(seg_frame/255)
                    self.control_car(action)

                    self.CarlaApi.tick()
                    reward,done = self.compute_reward()

                    next_rgb_frame, next_seg_frame = self.get_image()
    
                    self.ActorCritic.learn_critic(seg_frame/255, reward, next_seg_frame/255, done)
                    self.ActorCritic.learn_actor(seg_frame/255, action)

                    show_frame = np.hstack((seg_frame,rgb_frame))
                    cv2.imshow("",show_frame)
                    cv2.waitKey(1)

                self.CarlaApi.reset()
        finally:
            self.CarlaApi.destroy()
            print('destroy')
        
    """計算獎勵"""
    def compute_reward(self):
        sensor_data = self.CarlaApi.sensor_data()
        lane_line_info = sensor_data[0]
        collision_info = sensor_data[1]
        traffic_info = sensor_data[2]
        car_speed = sensor_data[3]

        reward = 0
        done = False
        print("car_speed:",car_speed)
        print(sensor_data)

        if(self.MIN_SPEED <= car_speed <= self.MAX_SPEED):
            speed_reward = 1.4 * (car_speed - self.MIN_SPEED)
        elif (car_speed < self.MIN_SPEED):
            speed_reward = 2.5 * (car_speed - self.MIN_SPEED)
        elif (car_speed > self.MAX_SPEED):
            speed_reward = 2 * (self.MAX_SPEED - car_speed)

        reward += speed_reward

        if lane_line_info:
            reward = -40
            done = True
        if collision_info:
            reward = -40
            done = True

        print('reward:',reward)

        return reward, done


    """車輛控制"""
    def control_car(self,action):
        """:param
                0:前進、1:煞車、2:半左轉、3:半右轉、4:全左轉、5:全右轉
        """
        control = carla.VehicleControl()
        control.throttle = 0.5
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


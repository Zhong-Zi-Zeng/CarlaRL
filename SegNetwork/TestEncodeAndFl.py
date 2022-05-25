from EncodeAndFlatten import Network
from CarlaApiAsync import CarlaApi
import cv2
import pygame
from pygame import K_UP
from pygame import K_DOWN
from pygame import K_LEFT
from pygame import K_RIGHT
from pygame import K_r
import numpy as np
import carla


# 載入權重
EncodeAndFlatten = Network(now_path='.').buildModel()
EncodeAndFlatten.model.build(input_shape=(75, 100, 64))
EncodeAndFlatten.load_weights()

# 初始化carla
CarlaApi = CarlaApi(img_width=400,img_height=300)
CarlaApi.initial(AutoMode=False)
CarlaApi.wait_for_sim()

# 初始化pygame
pygame.init()
clock = pygame.time.Clock()
win_screen = pygame.display.set_mode((400, 300))

control = carla.VehicleControl()
steer_cache = 0
def parseKeyControl(vehicle, keys, milliseconds):
    global steer_cache

    if keys[K_UP]:
        control.throttle = min(control.throttle + 0.01, 1.00)
    else:
        control.throttle = 0.0

    if keys[K_DOWN]:
        control.brake = min(control.brake + 0.2, 1)
    else:
        control.brake = 0

    steer_increment = 5e-4 * milliseconds
    if keys[K_LEFT]:
        if steer_cache > 0:
            steer_cache = 0
        else:
            steer_cache -= steer_increment
    elif keys[K_RIGHT]:
        if steer_cache < 0:
            steer_cache = 0
        else:
            steer_cache += steer_increment
    else:
        steer_cache = 0.0
    steer_cache = min(0.7, max(-0.7, steer_cache))
    control.steer = round(steer_cache, 1)
    vehicle.apply_control(control)


try:
    run = True
    while run:
        ori_frame = CarlaApi.camera_data()['front_bgr_camera']
        image_surface = pygame.surfarray.make_surface(ori_frame[:, :, ::-1].swapaxes(0, 1))
        win_screen.blit(image_surface, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    run = False
        parseKeyControl(CarlaApi.vehicle, pygame.key.get_pressed(), clock.get_time())


        pre_need_slow, pre_tl, pre_tl_dis= EncodeAndFlatten.predict(ori_frame)

        pre_need_slow = np.squeeze(pre_need_slow)
        pre_tl = np.squeeze(pre_tl)
        pre_tl_dis = np.squeeze(pre_tl_dis)

        if np.argmax(pre_need_slow) == 0:
            NeedSlow = 'True'
        else:
            NeedSlow = 'False'

        if np.argmax(pre_tl) == 0:
            Pre_TL = 'Green'
        elif np.argmax(pre_tl) == 1:
            Pre_TL = 'Red'
        else:
            Pre_TL = 'None'

        if np.argmax(pre_tl_dis) == 0:
            Pre_TL_dis = 'Close'
        elif np.argmax(pre_tl_dis) == 1:
            Pre_TL_dis = 'Medium'
        elif np.argmax(pre_tl_dis) == 2:
            Pre_TL_dis = 'Far'
        else:
            Pre_TL_dis = 'None'


        print('NeedSlow:',NeedSlow,'TL:',Pre_TL,'TL_dis:',Pre_TL_dis)

        pygame.display.update()

finally:
    CarlaApi.destroy()
    cv2.destroyAllWindows()
    print('Destroy actor')


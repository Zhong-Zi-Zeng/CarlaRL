import carla
import random
import cv2
import pygame
from pygame import K_UP
from pygame import K_DOWN
from pygame import K_LEFT
from pygame import K_RIGHT
from pygame import K_e
from pygame import K_f
import numpy as np
from collections import deque

# 初始化pygame
pygame.init()
clock = pygame.time.Clock()
win_screen = pygame.display.set_mode((400, 300))

# 繪製影像
def draw_image(bgr_frame):
    image_surface = pygame.surfarray.make_surface(bgr_frame[:,:,::-1].swapaxes(0, 1))
    win_screen.blit(image_surface,(0,0))

# 添加物件都會存於模擬環境中，當程式結束時需要刪除所有物件
# 不然下次再新增時可能會報錯誤
actor_list = []

# 與Sever端建立連接
client = carla.Client('localhost',2000)
client.set_timeout(30.0)

# 建立world物件，模擬環境裡的物件都是由這個來管理
world = client.get_world()
world.tick()
blueprint_library = world.get_blueprint_library()

map = world.get_map()

# ============車子============
bp = blueprint_library.filter('vehicle')[0]
transform = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(bp, transform)
vehicle.set_autopilot(False)
control = carla.VehicleControl()
steer_cache = 0.0
actor_list.append(vehicle)

# ============回調函式============
bgr_image_queue = deque(maxlen=5)
def callback(img):
    img = process_bgr_frame(img)
    bgr_image_queue.append(img)

# ============寫入label.txt============
count = 0
def writeLabel(img):
    global count
    if(count % 20 == 0):
        # print(count)
        with open('label.txt','a') as file:
            junction = '1' if need_slow else '0'
            tl = '1' if TL else '0'
            print('Write to txt file:',count)
            file.writelines(str(count) + ' ' + junction + ' ' + tl + '\n')

        cv2.imwrite('./data/%d.png'%(count),img)

    count += 1

# ============建立相機============
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '400')
camera_bp.set_attribute('image_size_y', '300')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(callback)


# ============影像轉ndarray============
def process_bgr_frame(bgr_frame):
    bgr_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(bgr_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (bgr_frame.height, bgr_frame.width, 4))
    bgr_frame = array[:, :, :3]

    return bgr_frame


# ============計算與waypoint的角度============
def countDegree(car_transform, waypoint):
    # car_transform = p.get_transform()
    car_mat = car_transform.get_matrix()
    car_dir = np.array([car_mat[0][0], car_mat[1][0]], dtype=np.float32)

    way_mat = waypoint.transform.get_matrix()
    way_dir = np.array([way_mat[0][3] - car_mat[0][3], way_mat[1][3] - car_mat[1][3]], dtype=np.float32)

    cos_theta = np.dot(car_dir, way_dir) / (np.linalg.norm(car_dir) * np.linalg.norm(way_dir))
    left_right = abs(np.cross(car_dir, way_dir)) / np.cross(car_dir, way_dir)

    cos_theta = np.clip(cos_theta, -1, 1)
    rad = np.arccos(cos_theta) * left_right

    degree = rad * 180 / np.pi

    return degree


# ============是否是需要減速的路段(岔路、轉彎路口等)============
def needSlow():
    way_list = []
    first = map.get_waypoint(vehicle.get_transform().location).next(3)[0]
    way_list.append(first)

    next_way = first.next(30)
    way_list += next_way

    for way in way_list[1:]:
        degree = countDegree(first.transform,way)
        if abs(degree) > 10:
            return True

    return False


# ============手動控制車輛============
def parseKeyControl(keys, milliseconds):
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

need_slow = 0
TL = 1
try:
    run = True
    flag = False
    while run:
        clock.tick_busy_loop(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_f:
                    if TL:
                        TL = 0
                    else:
                        TL = 1
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    run = False

        parseKeyControl(pygame.key.get_pressed(), clock.get_time())


        tlState = str(vehicle.get_traffic_light_state())
        if tlState == 'Red':
            flag = True
        elif tlState == 'Green' and TL == 0 and flag:
            flag = False
            TL = 1


        if len(bgr_image_queue):
            img = bgr_image_queue.pop()
            draw_image(img)
            pygame.display.update()

            need_slow_str = 'NeedSlow' if needSlow() else 'False'
            TL_str = 'Green' if TL else 'Red'
            print(need_slow_str, TL_str)
            # writeLabel(img)

finally:
    for actor in actor_list:
        actor.destroy()
    print('destroy actor')


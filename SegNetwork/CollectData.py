import carla
import random
import cv2
import pygame
from pygame import K_UP
from pygame import K_DOWN
from pygame import K_LEFT
from pygame import K_RIGHT
from pygame import K_f
import numpy as np
from collections import deque

# 初始化pygame
pygame.init()




# ============影像轉ndarray============
def process_bgr_frame(bgr_frame):
    bgr_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(bgr_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (bgr_frame.height, bgr_frame.width, 4))
    bgr_frame = array[:, :, :3]

    return bgr_frame


# ============計算與waypoint的角度============
def countDegree(car_transform, waypoint):
    if isinstance(car_transform,carla.Transform) and isinstance(waypoint,carla.Waypoint):
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
    else:
        raise ValueError('car_transform or waypoint is not correct object.')



class Collecter():
    def __init__(self,AutoMode=False):
        self.actor_list = []

        self.AutoMode = AutoMode
        self.bp = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.TL_list = None
        self.fileName = 0
        self.clock = pygame.time.Clock()
        self.win_screen = pygame.display.set_mode((400, 300))

        self.bgr_queue = deque(maxlen=5)
        self.initial()
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0


    """模擬環境初始化"""
    def initial(self):
        self._connect_to_world()
        self._search_world_tl()
        self._spawn_vehicle()
        self._spawn_camera()

    """查詢世界裡所有紅綠燈"""
    def _search_world_tl(self):
        self.TL_list = self.world.get_actors().filter("*traffic_light*")

    """連接到模擬環境"""
    def _connect_to_world(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)

        self.world = client.get_world()
        # self.world.tick()
        self.bp = self.world.get_blueprint_library()
        self.map = self.world.get_map()

    """繪製影像"""
    def draw_image(self,bgr_frame):
        image_surface = pygame.surfarray.make_surface(bgr_frame[:, :, ::-1].swapaxes(0, 1))
        self.win_screen.blit(image_surface, (0, 0))

    """產生車輛"""
    def _spawn_vehicle(self):
        bp = self.bp.filter('vehicle')[0]
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, transform)
        self.vehicle.set_autopilot(self.AutoMode)

        self.actor_list.append(self.vehicle)

    """產生相機"""
    def _spawn_camera(self):
        camera_bp = self.bp.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '400')
        camera_bp.set_attribute('image_size_y', '300')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(self._camera_callback)

        self.actor_list.append(camera)

    """相機回調函式"""
    def _camera_callback(self,img):
        img = process_bgr_frame(img)
        self.bgr_queue.append(img)

    """返回相機畫面"""
    def camera_data(self):
        if len(self.bgr_queue):
            return self.bgr_queue.pop()

        return

    """寫入label及儲存照片"""
    def writeLabel(self,img,need_slow,TL):
        if self.fileName % 5 == 0:
            with open('label.txt','a') as file:
                junction = '1' if need_slow else '0'
                tl = '1' if TL else '0'
                file.writelines(str(self.fileName) + ' ' + junction + ' ' + tl + '\n')
            cv2.imwrite('./data/%d.png'%(self.fileName),img)

        self.fileName += 1

    """判斷該路段是否需要減速"""
    def judge_need_slow(self):
        way_list = []
        first = self.map.get_waypoint(self.vehicle.get_transform().location).next(3)[0]
        way_list.append(first)

        next_way = first.next(30)
        way_list += next_way

        for way in way_list[1:]:
            degree = countDegree(first.transform, way)
            if abs(degree) > 10:
                return True

        return False

    """手動控制車輛"""
    def parseKeyControl(self,keys, milliseconds):
        if keys[K_UP]:
            self.control.throttle = min(self.control.throttle + 0.01, 1.00)
        else:
            self.control.throttle = 0.0

        if keys[K_DOWN]:
            self.control.brake = min(self.control.brake + 0.2, 1)
        else:
            self.control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT]:
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif keys[K_RIGHT]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = round(self.steer_cache, 1)
        self.vehicle.apply_control(self.control)

    """銷毀生成物件"""
    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()

# trafficLight = []
# for landmark in map.get_all_landmarks():
#     if landmark.is_dynamic:
#         trafficLight.append(landmark)
#
# for t in trafficLight:
#     print(t.waypoint)


collecter = Collecter(AutoMode=False)

try:
    run = True
    flag = False
    while run:
        collecter.clock.tick_busy_loop(60)

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

        collecter.parseKeyControl(pygame.key.get_pressed(), collecter.clock.get_time())
        img = collecter.camera_data()

        if img is not None:
            collecter.draw_image(img)


        pygame.display.update()


finally:
    collecter.destroy()
    print('destroy actor')


import carla
import random
import cv2
import pygame
import math
from pygame import K_UP
from pygame import K_DOWN
from pygame import K_LEFT
from pygame import K_RIGHT
from pygame import K_f
import numpy as np
from collections import deque




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

# ============查詢紅綠燈觸發框的位置============
def get_trafficlight_trigger_location(traffic_light):
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """
    def rotate_point(point, radians):
        """
        rotate a given point by a given angle
        """
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

        return carla.Vector3D(rotated_x, rotated_y, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent
    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)

# ============查詢物件與物件間的角度與距離是否在指定範圍內============
def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Check if a location is both within a certain distance from a reference object.
    By using 'angle_interval', the angle between the location and reference transform
    will also be tkaen into account, being 0 a location in front and 180, one behind.

    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    if norm_target > max_distance:
        return False

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return min_angle < angle < max_angle, norm_target, angle

class Collecter():
    def __init__(self,AutoMode=False):
        self.actor_list = []

        self.AutoMode = AutoMode
        self.bp = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.TL_list = None
        self.last_traffic_light = None
        self.fileName = 144520
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
        for tl in self.TL_list:
            tl.set_red_time(1.5)
            tl.set_yellow_time(0)
            tl.set_green_time(1.5)

    """連接到模擬環境"""
    def _connect_to_world(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)

        self.world = client.get_world()
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
        camera_transform = carla.Transform(carla.Location(x=1.5,y=0.4, z=2.4),carla.Rotation(yaw=-8))

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
    def writeLabel(self,img,need_slow,TL,TL_dis):
        if self.fileName % 5 == 0:
            with open('label.txt','a') as file:
                # print('now write {}'.format(self.fileName))
                file.writelines(str(self.fileName) + ' ' + need_slow + ' ' + TL + ' ' + TL_dis + '\n')
            cv2.imwrite('./data/%d.png'%(self.fileName),img)

        self.fileName += 1

    """判斷該路段是否需要減速"""
    def judge_need_slow(self):
        way_list = []
        first = self.map.get_waypoint(self.vehicle.get_transform().location).next(3)[0]
        way_list.append(first)

        next_way = first.next(15)
        way_list += next_way

        for way in way_list[1:]:
            degree = countDegree(first.transform, way)
            if abs(degree) > 10:
                return True

        return False

    """紅綠燈信號"""
    def affected_by_traffic_light(self):
        ego_vehicle_location = self.vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for traffic_light in self.TL_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self.map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            car_to_tl_info = is_within_distance(object_waypoint.transform, self.vehicle.get_transform(), 15, [0, 90])
            if not isinstance(car_to_tl_info,bool):
                if (car_to_tl_info[2] > 100 and car_to_tl_info[1] < 2.3) or \
                    car_to_tl_info[2] < 100:
                    return traffic_light.state, car_to_tl_info[1]
                else:
                    return 'dontShot'

        return None

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


pygame.init()
collecter = Collecter(AutoMode=True)

try:
    run = True
    while run:
        collecter.clock.tick_busy_loop(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    run = False

        collecter.parseKeyControl(pygame.key.get_pressed(), collecter.clock.get_time())
        img = collecter.camera_data()

        if img is not None:
            collecter.draw_image(img)
            tl = collecter.affected_by_traffic_light()
            need_slow = collecter.judge_need_slow()

            if tl != 'dontShot':
                if tl is not None:
                    # print('TL:',tl[0],'Dis:',tl[1],'needSlow:',need_slow)
                    # 紅綠燈hot code
                    if tl[0] == carla.TrafficLightState.Green:
                        TL = '1 0 0'
                    else:
                        TL = '0 1 0'
                    # 紅綠燈距離hot code
                    if tl[1] <= 5:
                        dis = '1 0 0 0'
                    elif tl[1] <= 10:
                        dis = '0 1 0 0'
                    else:
                        dis = '0 0 1 0'
                else:
                    TL = '0 0 1'
                    dis = '0 0 0 1'
                    # print('needSlow:',need_slow)
                need_slow = '1' if need_slow else '0'
                collecter.writeLabel(img,need_slow,TL,dis)

        pygame.display.update()

finally:
    collecter.destroy()
    print('destroy actor')


import os
import carla
import random
import cv2
import pygame
import math
import queue
from pygame import K_UP
from pygame import K_DOWN
from pygame import K_LEFT
from pygame import K_RIGHT
from pygame import K_r
from pygame import K_q
import numpy as np
from collections import deque



# ============影像轉ndarray============
def process_bgr_frame(bgr_frame):
    bgr_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(bgr_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (bgr_frame.height, bgr_frame.width, 4))
    bgr_frame = array[:, :, :3]

    return bgr_frame

def process_seg_img(seg_frame):
    seg_frame.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(seg_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (seg_frame.height, seg_frame.width, 4))
    seg_frame = array[:, :, :3]

    return seg_frame

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

class Collector():
    def __init__(self,sys_mode,Label,AutoMode=False):
        self.actor_list = []
        self.sensor_list = []
        self.queues = []
        self.label = Label
        self.AutoMode = AutoMode
        self.sys_mode = sys_mode
        self.bp = None
        self.world = None
        self.tm = None
        self.map = None
        self.vehicle = None
        self.TL_list = None
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0

        if self.label == 'Seg':
            self.fileName = os.listdir('./seg')
        else:
            self.fileName = os.listdir('./data')
        self.fileName.sort(key=lambda x:int(x[:-4]))
        self.fileName.reverse()
        if len(self.fileName) == 0:
            self.fileName = 0
        else:
            self.fileName = int(self.fileName[0][:-4]) + 1

        self.clock = pygame.time.Clock()
        self.win_screen = pygame.display.set_mode((400, 300))
        self.initial()

    """模擬環境初始化"""
    def initial(self):
        self._connect_to_world()
        self._search_world_tl()
        self._spawn_vehicle()
        self._spawn_camera()
        self._make_event()

    """查詢世界裡所有紅綠燈"""
    def _search_world_tl(self):
        self.TL_list = self.world.get_actors().filter("*traffic_light*")

    """設定紅綠燈時間"""
    def set_tl_state_time(self):
        for tl in self.TL_list:
            tl.set_red_time(0)
            tl.set_yellow_time(0)
            tl.set_green_time(4)

    """連接到模擬環境"""
    def _connect_to_world(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)
        self.world = client.get_world()
        self.bp = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        if self.sys_mode:
            self._settings = self.world.get_settings()
            self.frame = self.world.apply_settings(carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=1/20))
        
    """同步模式"""
    def tick(self):
        def check_data(q):
            data = q.get(timeout=4.0)
            if data.frame == self.frame:
                return data

        self.frame = self.world.tick()
        camera_data = [check_data(sensor_queue) for sensor_queue in self.queues]

        return camera_data

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
        if self.AutoMode:
            self.vehicle.enable_constant_velocity(carla.Vector3D(x=2.7))
        self.actor_list.append(self.vehicle)

    """隨機移動車輛位置"""
    def random_move_vehicle(self):
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle.set_transform(transform)

    """產生相機"""
    def _spawn_camera(self):
        camera_bp = self.bp.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '400')
        camera_bp.set_attribute('image_size_y', '300')
        camera_transform = carla.Transform(carla.Location(x=1.5, y=-0, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        self.actor_list.append(camera)
        self.sensor_list.append(camera)

        seg_camera_bp = self.bp.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '400')
        seg_camera_bp.set_attribute('image_size_y', '300')
        seg_camera_transform = carla.Transform(carla.Location(x=1.5, y=0, z=2.4))
        seg_camera = self.world.spawn_actor(seg_camera_bp, seg_camera_transform, attach_to=self.vehicle)

        self.actor_list.append(seg_camera)
        self.sensor_list.append(seg_camera)

    """監聽"""
    def _make_event(self):
        def make_queue(s):
            q = queue.Queue()
            s.listen(q.put)
            self.queues.append(q)

        for sensor in self.sensor_list:
            make_queue(sensor)

    """寫入label及儲存照片"""
    def writeLabel(self,bgr_img,seg_img):
        if self.label == 'Seg':
            cv2.imwrite('./ori/%d.png' % (self.fileName), bgr_img)
            cv2.imwrite('./seg/%d.png' % (self.fileName), seg_img)
            self.fileName += 1
        else:
            tl = self.affected_by_traffic_light()
            need_slow = self.judge_need_slow()

            if tl != 'dontShot':
                if tl is not None:
                    # 紅綠燈hot code
                    if tl[0] == carla.TrafficLightState.Green:
                        TL = '1 0 0'
                    elif tl[0] == carla.TrafficLightState.Red:
                        TL = '0 1 0'
                    else:
                        TL = '0 0 1'
                    # 紅綠燈距離hot code
                    if tl[1] <= 5:
                        TL_dis = '1 0 0 0'
                    elif tl[1] <= 10:
                        TL_dis = '0 1 0 0'
                    else:
                        TL_dis = '0 0 1 0'
                else:
                    TL = '0 0 1'
                    TL_dis = '0 0 0 1'

                need_slow = '1 0' if need_slow else '0 1'

                with open('label.txt','a') as file:
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

        if keys[K_q]:
            self.control.gear = 1 if self.control.reverse else -1
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = round(self.steer_cache, 1)
        self.vehicle.apply_control(self.control)

    """銷毀生成物件"""
    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()


pygame.init()
SYS_MODE = True
collector = Collector(AutoMode=False, sys_mode=SYS_MODE,Label='Seg')

try:
    run = True
    while run:
        collector.clock.tick_busy_loop(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    run = False

        img = collector.tick()
        bgr_img = process_bgr_frame(img[0])
        seg_img = process_seg_img(img[1])

        collector.draw_image(bgr_img)
        collector.set_tl_state_time()
        keys = pygame.key.get_pressed()
        if keys[K_r]:
            collector.writeLabel(bgr_img, seg_img)
        # collector.writeLabel(bgr_img, seg_img)
        # collector.random_move_vehicle()

        collector.parseKeyControl(pygame.key.get_pressed(), collector.clock.get_time())

        pygame.display.update()
finally:
    collector.destroy()
    print('destroy actor')


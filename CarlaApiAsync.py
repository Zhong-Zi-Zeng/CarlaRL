import carla
import numpy as np
from queue import Queue
from collections import deque
import time


def process_bgr_frame(bgr_frame):
    bgr_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(bgr_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (bgr_frame.height, bgr_frame.width, 4))
    bgr_frame = array[:, :, :3]

    return bgr_frame

def process_seg_frame(seg_frame):
    seg_frame.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(seg_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (seg_frame.height, seg_frame.width, 4))
    seg_frame = array[:, :, :3]

    return seg_frame

class CarlaApi:
    def __init__(self,img_width,img_height):
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.vehicle_transform = None
        self.block = False
        self.img_width = img_width
        self.img_height = img_height
        self.frame = 0

        self.sensor_list = []
        self.camera_list = []
        self.sensor_queue_list = []
        self.camera_queue_list = []
        self.sensor_info_queue = deque(maxlen=5)
        self.camera_info_queue = deque(maxlen=5)

    """連接到模擬環境"""
    def connect_to_world(self):
        print('Connect to world....')
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        self.world = client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

    """取出佇列資料"""
    def pop_queue(self,Q):
        data = Q.pop() if len(Q) else None
        return data

    """on_tick的回調函式"""
    def _callback(self,WorldSnapshot):
        self._camera_callback()
        self._sensor_callback()

    """相機回調函式"""
    def _camera_callback(self):
        camera_info = {camera_name : self.pop_queue(camera_queue) \
                       for camera_queue, camera_name in self.camera_queue_list}

        self.camera_info_queue.append(camera_info)

    """感測器回調函式"""
    def _sensor_callback(self):
        if not self.block:
            sensor_info = {sensor_name: self.pop_queue(sensor_queue) \
                           for sensor_queue, sensor_name in self.sensor_queue_list}

            self.sensor_info_queue.append(sensor_info)

            if sensor_info['lane_line_sensor'] or sensor_info['collision_sensor']:
                self._spawn_vehicle()
                self.block = True

    """等待模擬開始"""
    def wait_for_sim(self):
        self.world.wait_for_tick()

    """產生車輛"""
    def _spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('vehicle')[0]
        if(self.vehicle_transform is not None):
            # self.vehicle_transform = np.random.choice(self.world.get_map().get_spawn_points())
            self.vehicle.set_transform(self.vehicle_transform)
        else:
            self.vehicle_transform = np.random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.vehicle_transform)
        self.vehicle.set_autopilot(False)

    """產生攝影機"""
    def _spawn_camera(self):
        """產生bgr攝影機到車上"""
        bgr_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        bgr_camera_bp.set_attribute('image_size_x', str(self.img_width))
        bgr_camera_bp.set_attribute('image_size_y', str(self.img_height))
        bgr_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        bgr_camera = self.world.spawn_actor(bgr_camera_bp, bgr_camera_transform, attach_to=self.vehicle)

        self.camera_list.append([bgr_camera, 'bgr_camera'])

        """產生SEG攝影機到車上"""
        seg_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', str(self.img_width))
        seg_camera_bp.set_attribute('image_size_y', str(self.img_height))
        seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        seg_camera = self.world.spawn_actor(seg_camera_bp, seg_camera_transform, attach_to=self.vehicle)

        self.camera_list.append([seg_camera, 'seg_camera'])

    """產生感測器"""
    def _spawn_sensor(self):
        """產生車道偵測感測器"""
        lane_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        lane_line_sensor = self.world.spawn_actor(lane_sensor_bp, carla.Transform(), attach_to=self.vehicle)

        self.sensor_list.append([lane_line_sensor,'lane_line_sensor'])

        """產生碰撞感測器"""
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)

        self.sensor_list.append([collision_sensor,'collision_sensor'])

    """建立佇列"""
    def _build_queue(self):
        for sensor, sensor_name in self.sensor_list:
            Q = deque(maxlen=5)
            sensor.listen(Q.append)
            self.sensor_queue_list.append([Q,sensor_name])

        for camera, camera_name in self.camera_list:
            Q = deque(maxlen=5)
            camera.listen(Q.append)
            self.camera_queue_list.append([Q,camera_name])

    """清空佇列"""
    def _clear_queue(self):
        self.sensor_info_queue.clear()
        self.camera_info_queue.clear()

        for camera_queue, camera_name in self.camera_queue_list:
            camera_queue.clear()

        for sensor_queue, sensor_name in self.sensor_queue_list:
            sensor_queue.clear()

    """銷毀生成物件"""
    def destroy(self):
        self.vehicle.destroy()
        for actor in self.sensor_list:
            actor[0].destroy()

    """初始化"""
    def initial(self):
        self.connect_to_world()
        self._spawn_vehicle()
        self._spawn_sensor()
        self._spawn_camera()
        self._spawn_finish_point()
        self._build_queue()
        self.world.on_tick(self._callback)

    """重置"""
    def reset(self):
        velocity = carla.Vector3D(x=0.0,y=0.0,z=0.0)
        self.vehicle.set_target_velocity(velocity)
        self.block = False
        self._spawn_finish_point()
        self._clear_queue()

    """控制車子"""
    def control_vehicle(self,control):
        if isinstance(control,carla.VehicleControl):
            self.vehicle.apply_control(control)
        else:
            print('The parameter "control" must be carla.VehicleControl object.')

    """取得道路中心點座標"""
    def _spawn_finish_point(self):
        self.way_point = self.world.get_map().get_waypoint(self.vehicle_transform.location)
        self.way_point = self.way_point.next(20)

    """返回攝影機數據"""
    def camera_data(self):
        while True:
            try:
                camera_info = self.camera_info_queue.pop()
                camera_info['bgr_camera'] = process_bgr_frame(camera_info['bgr_camera'])
                camera_info['seg_camera'] = process_seg_frame(camera_info['seg_camera'])

                return camera_info
            except:
                continue

    """返回感測器數據"""
    def sensor_data(self):
        while True:
            try:
                # 交通號誌訊息
                sensor_info = self.sensor_info_queue.pop()
                sensor_info['traffic_info'] = self.vehicle.get_traffic_light_state()

                # 車輛速度
                car_speed = self.vehicle.get_velocity()
                car_speed = np.sqrt(car_speed.x ** 2 + car_speed.y ** 2 + car_speed.z ** 2) * 3.6
                sensor_info['car_speed'] = car_speed

                # 與目標點距離
                car_location = self.vehicle.get_location()
                target_location = self.way_point[0].transform.location
                sensor_info['finish_point_dis'] = np.sqrt((car_location.x - target_location.x) ** 2 +
                                                          (car_location.y - target_location.y) ** 2 +
                                                          (car_location.z - target_location.z) ** 2)
                if(sensor_info['finish_point_dis'] < 3):
                    self.way_point = self.way_point.next(20)

                return sensor_info
            except:
                continue

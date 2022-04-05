from re import L
import carla
import numpy as np
from queue import Queue
import cv2
from threading import Thread

class CarlaApi:
    def __init__(self,host,image_width,image_height,queue_max_size,fps):
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.vehicle_transform = None
        self.x_frame = None

        self.HOST = host
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.MAX_SIZE = queue_max_size
        self.FPS = fps

        self.sensor_list = []

    """初始化"""
    def initial(self):
        self.connect_to_world()
        self._build_queue()
        self._spawn_vehicle()
        self._spawn_sensor()

    """build queue"""
    def _build_queue(self):
        self.seg_image_queue = Queue(maxsize=self.MAX_SIZE)
        self.rgb_image_queue = Queue(maxsize=self.MAX_SIZE)
        self.lane_line_info_queue = Queue(maxsize=self.MAX_SIZE)
        self.traffic_light_info_queue = Queue(maxsize=self.MAX_SIZE)
        self.collision_info_queue = Queue(maxsize=self.MAX_SIZE)

    """連接到模擬環境"""
    def connect_to_world(self):
        print('Connect to world....')
        client = carla.Client(self.HOST, 2000)
        client.set_timeout(30.0)
        self.world = client.get_world()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.FPS))
        self.blueprint_library = self.world.get_blueprint_library()

    """產生車輛"""
    def _spawn_vehicle(self):
        bp = self.blueprint_library.filter('vehicle')[0]
        if(self.vehicle_transform is not None):
            self.vehicle.set_transform(self.vehicle_transform)
        else:
            self.vehicle_transform = np.random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(bp, self.vehicle_transform)
        self.vehicle.set_autopilot(False)

    """產生感測器"""
    def _spawn_sensor(self):
        """產生RGB攝影機到車上"""
        bgr_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        bgr_camera_bp.set_attribute('image_size_x', str(self.IMAGE_WIDTH))
        bgr_camera_bp.set_attribute('image_size_y', str(self.IMAGE_HEIGHT))
        bgr_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        bgr_camera = self.world.spawn_actor(bgr_camera_bp, bgr_camera_transform, attach_to=self.vehicle)
        bgr_camera.listen(self.rgb_image_queue.put)

        self.sensor_list.append(bgr_camera)

        """產生SEG攝影機到車上"""
        seg_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', str(self.IMAGE_WIDTH))
        seg_camera_bp.set_attribute('image_size_y', str(self.IMAGE_HEIGHT))
        seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        seg_camera = self.world.spawn_actor(seg_camera_bp, seg_camera_transform, attach_to=self.vehicle)
        seg_camera.listen(self.seg_image_queue.put)

        self.sensor_list.append(seg_camera)

        """產生車道偵測感測器"""
        lane_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        lane_line_sensor = self.world.spawn_actor(lane_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        lane_line_sensor.listen(self.lane_line_info_queue.put)

        self.sensor_list.append(lane_line_sensor)

        """產生碰撞感測器"""
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        collision_sensor.listen(self.collision_info_queue.put)

        self.sensor_list.append(collision_sensor)

    """銷毀生成物件"""
    def _destroy(self):
        self.vehicle.destroy()
        for sensor in self.sensor_list:
            sensor.destroy()
        self.sensor_list = []

    """控制車子"""
    def control_vehicle(self,control):
        if isinstance(control,carla.VehicleControl):
            self.vehicle.apply_control(control)
        else:
            print('The parameter "control" must be carla.VehicleControl object.')

    """重置"""
    def reset(self):
        control = carla.VehicleControl()
        control.throttle = 0.5
        control.steer = 0.0
        self.control_vehicle(control)
        self._spawn_vehicle()

    """返回攝影機資料"""
    def camera_data(self,timeout):
        """處理rgb畫面"""
        def process_rgb_frame(rgb_frame):
            rgb_frame.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(rgb_frame.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (rgb_frame.height, rgb_frame.width, 4))
            rgb_frame = array[:, :, :3]

            return rgb_frame

        """處理seg畫面"""
        def process_seg_frame(seg_frame):
            seg_frame.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(seg_frame.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (seg_frame.height, seg_frame.width, 4))
            seg_frame = array[:, :, :3]

            return seg_frame

        rgb_info = self.rgb_image_queue.get(timeout=timeout)
        rgb_frame = process_rgb_frame(rgb_info)
        seg_info = self.seg_image_queue.get(timeout=timeout)
        seg_frame = process_seg_frame(seg_info)

        return rgb_frame, seg_frame

    """進行模擬"""
    def tick(self):
        self.x_frame = self.world.tick()

    """返回感測器資料"""
    def sensor_data(self):
        lane_line_info = None
        collision_info = None

        # if not self.lane_line_info_queue.empty():
        while not self.lane_line_info_queue.empty():
            lane_line_info = self.lane_line_info_queue.get()
            lane_line_info = True if lane_line_info.frame == self.x_frame else None
            if lane_line_info:
                break

        # if not self.collision_info_queue.empty():
        while not self.collision_info_queue.empty():
            collision_info = self.collision_info_queue.get()
            collision_info = True if collision_info == self.x_frame else None
            if collision_info:
                break

        traffic_light_info = self.vehicle.get_traffic_light_state()

        car_speed = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_speed.x**2 + car_speed.y**2 + car_speed.z**2) * 3.6

        return [lane_line_info, collision_info, traffic_light_info, car_speed]

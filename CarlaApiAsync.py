import carla
import numpy as np
from queue import Queue
import cv2


class CarlaSyncMode(object):
    def __init__(self, world, sensor_list):
        self.world = world
        self.sensor_list = sensor_list
        self.frame = None
        self._queues = []

    def make_event_queue(self):
        def make_queue(register_event):
            q = Queue()
            register_event(q.put)
            self._queues.append(q)

        for sensor in self.sensor_list:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            print(data)
            if data.frame == self.frame:
                return data


def process_rgb_frame(rgb_frame):
    rgb_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(rgb_frame.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (rgb_frame.height, rgb_frame.width, 4))
    rgb_frame = array[:, :, :3]

    return rgb_frame

def process_seg_frame(seg_frame):
    seg_frame.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(seg_frame.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (seg_frame.height, seg_frame.width, 4))
    seg_frame = array[:, :, :3]

    return seg_frame


class CarlaApi:
    def __init__(self):
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.vehicle_transform = None
        self.frame = None
        self.FPS = 30
        self.sensor_list = []

    """連接到模擬環境"""
    def connect_to_world(self):
        print('Connect to world....')
        host = '192.168.88.251'
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        self.world = client.get_world()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.FPS,
            ))

        self.blueprint_library = self.world.get_blueprint_library()

    """產生車輛"""
    def _spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('vehicle')[0]
        if(self.vehicle_transform is not None):
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.vehicle_transform)
        else:
            self.vehicle_transform = np.random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.vehicle_transform)
        self.vehicle.set_autopilot(False)


    """產生感測器"""
    def _spawn_sensor(self):
        """產生RGB攝影機到車上"""
        bgr_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        bgr_camera_bp.set_attribute('image_size_x', '100')
        bgr_camera_bp.set_attribute('image_size_y', '100')
        bgr_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        bgr_camera = self.world.spawn_actor(bgr_camera_bp, bgr_camera_transform, attach_to=self.vehicle)

        self.sensor_list.append(bgr_camera)

        """產生SEG攝影機到車上"""
        seg_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '100')
        seg_camera_bp.set_attribute('image_size_y', '100')
        seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        seg_camera = self.world.spawn_actor(seg_camera_bp, seg_camera_transform, attach_to=self.vehicle)

        self.sensor_list.append(seg_camera)

        """產生車道偵測感測器"""
        lane_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        lane_line_sensor = self.world.spawn_actor(lane_sensor_bp, carla.Transform(), attach_to=self.vehicle)

        self.sensor_list.append(lane_line_sensor)

        """產生碰撞感測器"""
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)

        self.sensor_list.append(collision_sensor)

    """銷毀生成物件"""
    def destroy(self):
        self.sensor_list.append(self.vehicle)
        for actor in self.sensor_list:
            actor.destroy()

    """初始化"""
    def initial(self):
        self.connect_to_world()
        self._spawn_vehicle()
        self._spawn_sensor()

        self.CarlaSyncMode = CarlaSyncMode(self.world,self.sensor_list)
        self.CarlaSyncMode.make_event_queue()

    """重置"""
    def reset(self):
        self._spawn_vehicle()

    """控制車子"""
    def control_vehicle(self,control):
        if isinstance(control,carla.VehicleControl):
            self.vehicle.apply_control(control)
        else:
            print('The parameter "control" must be carla.VehicleControl object.')

    """返回感測器資料"""
    def sensor_data(self):
        rgb_frame, seg_frame, lane_line_info, collision_info = self.CarlaSyncMode.tick(None)
        traffic_light_info = self.vehicle.get_traffic_light_state()
        car_speed = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_speed.x**2 + car_speed.y**2 + car_speed.z**2) * 3.6

        return [rgb_frame, seg_frame, lane_line_info, collision_info, traffic_light_info, car_speed]



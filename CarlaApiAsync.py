import carla
import numpy as np
from queue import Queue


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

        self.sensor_list = []
        self.sensor_queue_list = []
        self.sensor_info = {}

    """連接到模擬環境"""
    def connect_to_world(self):
        print('Connect to world....')
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        self.world = client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

    """on_tick的回調函式"""
    def _callback(self,WorldSnapshot):
        world_frame = WorldSnapshot.frame
        self.sensor_info = {}
        print('world:',world_frame)
        def check_frame(sensor_queue):
            while True:
                try:
                    data = sensor_queue.get(timeout=0.3)
                    print('data:',data.frame)
                    if data.frame == world_frame:
                        return data
                except:
                    return

        self.sensor_info = {sensor_name : check_frame(sensor_queue) \
                            for sensor_queue,sensor_name in self.sensor_queue_list}
        print(self.sensor_info)

    """產生車輛"""
    def _spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('vehicle')[0]
        if(self.vehicle_transform is not None):
            self.vehicle = self.vehicle.set_transform(self.vehicle_transform)
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

        self.sensor_list.append([bgr_camera,'bgr_camera'])

        """產生SEG攝影機到車上"""
        seg_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '100')
        seg_camera_bp.set_attribute('image_size_y', '100')
        seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        seg_camera = self.world.spawn_actor(seg_camera_bp, seg_camera_transform, attach_to=self.vehicle)

        self.sensor_list.append([seg_camera,'seg_camera'])

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
            Q = Queue()
            sensor.listen(Q.put)
            self.sensor_queue_list.append([Q,sensor_name])

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
        self._build_queue()
        self.world.on_tick(self._callback)

    """重置"""
    def reset(self):
        control = carla.VehicleControl()
        control.throttle = 0.5
        control.steer = 0.0
        self._spawn_vehicle()

    """控制車子"""
    def control_vehicle(self,control):
        if isinstance(control,carla.VehicleControl):
            self.vehicle.apply_control(control)
        else:
            print('The parameter "control" must be carla.VehicleControl object.')

    """返回感測器數據"""
    def sensor_data(self):
        while True:
            if self.sensor_info:
                self.sensor_info['bgr_camera'] = process_rgb_frame(self.sensor_info['bgr_camera'])
                self.sensor_info['seg_camera'] = process_seg_frame(self.sensor_info['seg_camera'])
                self.sensor_info['traffic_info'] = self.vehicle.get_traffic_light_state()

                car_speed = self.vehicle.get_velocity()
                car_speed = np.sqrt(car_speed.x ** 2 + car_speed.y ** 2 + car_speed.z ** 2) * 3.6
                self.sensor_info['car_speed'] = car_speed

                return self.sensor_info




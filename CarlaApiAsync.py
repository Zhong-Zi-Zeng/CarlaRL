import carla
import numpy as np
import math
from collections import deque

# ============處理bgr影像============
def process_bgr_frame(bgr_frame):
    bgr_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(bgr_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (bgr_frame.height, bgr_frame.width, 4))
    bgr_frame = array[:, :, :3]

    return bgr_frame

# ============處理seg影像============
def process_seg_frame(seg_frame):
    seg_frame.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(seg_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (seg_frame.height, seg_frame.width, 4))
    seg_frame = array[:, :, :3]

    return seg_frame

# ============計算與waypoint的角度============
def countDegree(car_transform, waypoint):
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

class CarlaApi:
    def __init__(self,img_width,img_height,MIN_MIDDLE_DIS=0.6):
        self.world = None
        self.map = None
        self.blueprint_library = None
        self.vehicle = None
        self.vehicle_transform = None
        self.TL_list = None
        self.block = False
        self.MIN_MIDDLE_DIS = MIN_MIDDLE_DIS
        self.img_width = img_width
        self.img_height = img_height
        self.frame = 0

        self.waypoint_list = []
        self.sensor_list = []
        self.camera_list = []
        self.sensor_queue_list = []
        self.camera_queue_list = []
        self.sensor_info_queue = deque(maxlen=5)
        self.camera_info_queue = deque(maxlen=5)

    """連接到模擬環境"""
    def _connect_to_world(self):
        print('Connect to world....')
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

    """查詢世界裡所有紅綠燈"""
    def _search_world_tl(self):
        self.TL_list = self.world.get_actors().filter("*traffic_light*")

    """取出佇列資料"""
    def pop_queue(self,Q):
        data = Q.pop() if len(Q) else None
        return data

    """on_tick的回調函式"""
    def _callback(self,WorldSnapshot):
        self._camera_callback()
        self._sensor_callback()
        self._next_waypoint_callback()

    """是否切換下一個路徑點"""
    def _next_waypoint_callback(self):
        way = self.waypoint_list[0]
        dis = self.vehicle.get_transform().location.distance(way.transform.location)
        if dis < self.MIN_MIDDLE_DIS:
            self._toggle_waypoint()

    """相機回調函式"""
    def _camera_callback(self):
        camera_info = {camera_name : self.pop_queue(camera_queue) \
                       for camera_queue, camera_name in self.camera_queue_list}

        self.camera_info_queue.append(camera_info)

    """感測器回調函式"""
    def _sensor_callback(self):
        # if not self.block:
        sensor_info = {sensor_name: self.pop_queue(sensor_queue) \
                       for sensor_queue, sensor_name in self.sensor_queue_list}

        self.sensor_info_queue.append(sensor_info)
            # self.block = True if sensor_info['collision_sensor'] else False

    """等待模擬開始"""
    def wait_for_sim(self):
        self.world.wait_for_tick()

    """切換下一個路徑點"""
    def _toggle_waypoint(self):
        self.waypoint_list.pop(0)
        # 當路徑點被取完後生成新的路徑點
        if not len(self.waypoint_list):
            self._build_waypoint()

    """產生車輛"""
    def _spawn_vehicle(self,AutoMode):
        vehicle_bp = self.blueprint_library.filter('vehicle')[0]
        if(self.vehicle_transform is not None):
            self.vehicle.set_transform(self.vehicle_transform)
        else:
            self.vehicle_transform = np.random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.vehicle_transform)

        self.vehicle.set_autopilot(AutoMode)

    """產生攝影機"""
    def _spawn_camera(self):
        """產生前方bgr攝影機"""
        front_bgr_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        front_bgr_camera_bp.set_attribute('image_size_x', str(self.img_width))
        front_bgr_camera_bp.set_attribute('image_size_y', str(self.img_height))
        front_bgr_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        front_bgr_camera = self.world.spawn_actor(front_bgr_camera_bp, front_bgr_camera_transform, attach_to=self.vehicle)

        self.camera_list.append([front_bgr_camera, 'front_bgr_camera'])

        """產生上方bgr攝影機"""
        top_bgr_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        top_bgr_camera_bp.set_attribute('image_size_x', str(self.img_width * 2))
        top_bgr_camera_bp.set_attribute('image_size_y', str(self.img_height * 2))
        top_bgr_camera_transform = carla.Transform(carla.Location(x=-5, z=4),carla.Rotation(pitch=-15.0))
        top_bgr_camera = self.world.spawn_actor(top_bgr_camera_bp, top_bgr_camera_transform, attach_to=self.vehicle)

        self.camera_list.append([top_bgr_camera, 'top_bgr_camera'])

        """產生SEG攝影機到車上"""
        seg_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', str(self.img_width))
        seg_camera_bp.set_attribute('image_size_y', str(self.img_height))
        seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        seg_camera = self.world.spawn_actor(seg_camera_bp, seg_camera_transform, attach_to=self.vehicle)

        self.camera_list.append([seg_camera, 'seg_camera'])

    """產生感測器"""
    def _spawn_sensor(self):
        """產生碰撞感測器"""
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)

        self.sensor_list.append([collision_sensor,'collision_sensor'])

    """生成路徑點"""
    def _build_waypoint(self,distance=10000,sample=3):
        self.waypoint_list = []
        first_way = self.map.get_waypoint(self.vehicle_transform.location).next(sample)
        self.waypoint_list.append(first_way[0])

        for i in range(distance // sample):
            way = self.waypoint_list[i].next(sample)
            self.waypoint_list.append(way[0])

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
    def initial(self,AutoMode=False):
        self._connect_to_world()
        self._search_world_tl()
        self._spawn_vehicle(AutoMode)
        self._spawn_sensor()
        self._spawn_camera()
        self._build_queue()
        self._build_waypoint()
        self.world.on_tick(self._callback)

    """重置"""
    def reset(self):
        velocity = carla.Vector3D(x=0.0,y=0.0,z=0.0)
        control = carla.VehicleControl(throttle=0.0,steer=0.0,brake=0)
        self.vehicle.set_target_velocity(velocity)
        self.control_vehicle(control)
        self.block = False
        self._spawn_vehicle(AutoMode=False)
        self._build_waypoint()
        self._clear_queue()

    """控制車子"""
    def control_vehicle(self,control):
        if isinstance(control,carla.VehicleControl):
            self.vehicle.apply_control(control)
        else:
            print('The parameter "control" must be carla.VehicleControl object.')

    """返回攝影機數據"""
    def camera_data(self):
        while True:
            try:
                camera_info = self.camera_info_queue.pop()
                camera_info['front_bgr_camera'] = process_bgr_frame(camera_info['front_bgr_camera'])
                camera_info['top_bgr_camera'] = process_bgr_frame(camera_info['top_bgr_camera'])
                camera_info['seg_camera'] = process_seg_frame(camera_info['seg_camera'])

                return camera_info
            except:
                continue

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
            if not isinstance(car_to_tl_info, bool):
                if (car_to_tl_info[2] > 100 and car_to_tl_info[1] > 2.3):
                    return True
        return False

    """獲取車輛訊息"""
    def car_data(self):
        car_info = {}

        # 車輛速度
        car_speed = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_speed.x ** 2 + car_speed.y ** 2 + car_speed.z ** 2) * 3.6
        car_info['car_speed'] = car_speed

        # 方向盤數值
        # steering = self.vehicle.get_control().steer
        # car_info['car_steering'] = steering

        # 車輛與當前路徑點的角度差與距離
        way = self.waypoint_list[0]
        way_dis = way.transform.location.distance(self.vehicle.get_transform().location)
        car_info['way_dis'] = way_dis

        way_angel = countDegree(self.vehicle.get_transform(),way)
        car_info['way_degree'] = way_angel

        # 紅綠燈訊息
        car_info['tl'] = self.affected_by_traffic_light()

        return car_info

    """返回感測器數據"""
    def sensor_data(self):
        while True:
            try:
                sensor_info = self.sensor_info_queue.pop()
                return sensor_info
            except:
                continue

import random
import pygame
import cv2
import carla
import queue
import numpy as np

class Collector:
    def __init__(self):
        self.actor_list = []
        self.sensor_list = []
        self.queues = []
        self.vehicle = None
        self.world = None
        self.bp = None
        self.frame = None

    """初始化"""
    def initial(self):
        self._connect_to_world()
        self._spawn_vehicle()
        self._spawn_camera()
        self._make_event()

    """連接到模擬環境"""
    def _connect_to_world(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)
        self.world = client.get_world()
        self.bp = self.world.get_blueprint_library()

        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1 / 20))

    """隨機移動車輛位置"""
    def random_move_vehicle(self):
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle.set_transform(transform)

    """產生車輛"""
    def _spawn_vehicle(self):
        bp = self.bp.filter('vehicle')[0]
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, transform)
        self.vehicle.set_autopilot(False)

        self.actor_list.append(self.vehicle)

    """產生攝影機"""
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

    """同步"""
    def tick(self):
        def check_data(q):
            data = q.get(timeout=4.0)
            if data.frame == self.frame:
                return data

        self.frame = self.world.tick()
        camera_data = [check_data(sensor_queue) for sensor_queue in self.queues]

        return camera_data

    """監聽"""
    def _make_event(self):
        def make_queue(s):
            q = queue.Queue()
            s.listen(q.put)
            self.queues.append(q)

        for sensor in self.sensor_list:
            make_queue(sensor)

    """銷毀生成物件"""
    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()

def process_img(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    return array

def process_seg_img(seg_frame):
    seg_frame.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(seg_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (seg_frame.height, seg_frame.width, 4))
    seg_frame = array[:, :, :3]

    return seg_frame

collector = Collector()
collector.initial()
pygame.init()
win_screen = pygame.display.set_mode((400, 300))
try:
    run = True
    file_name = 0
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    run = False

        bgr_data, seg_data = collector.tick()
        bgr_img = process_img(bgr_data)
        seg_img = process_seg_img(seg_data)

        cv2.imwrite('./ori/%s.png'%(file_name),bgr_img)
        cv2.imwrite('./seg/%s.png' % (file_name),seg_img)
        file_name += 1
        collector.random_move_vehicle()
finally:
    collector.destroy()
    print('destroy actor')
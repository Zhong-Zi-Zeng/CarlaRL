import carla
from carla.libcarla import ActorBlueprint, Vehicle
import random
import queue
import numpy as np
import cv2



# 添加物件都會存於模擬環境中，當程式結束時需要刪除所有物件
# 不然下次再新增時可能會報錯誤
actor_list = []

# try:
# 與Sever端建立連接
client = carla.Client('localhost',2000)
client.set_timeout(30.0)

# 建立world物件，模擬環境裡的物件都是由這個來管理
world = client.get_world()
# world.tick()

blueprint_library = world.get_blueprint_library()

# ============車子============

# 創建一台車子型號
bp = random.choice(blueprint_library.filter('vehicle'))

# 隨機初始化位置
transform = random.choice(world.get_map().get_spawn_points())

# 新增車子至模擬環境中
vehicle = world.spawn_actor(bp, transform)

# 是否將車子設為自動駕駛
vehicle.set_autopilot(True)

actor_list.append(vehicle)

# ============控制車子============
# car_control = carla.VehicleControl()
# car_control.throttle = 0.5
# car_control.steer = 0.3
# vehicle.apply_control(car_control)

# ============車道線偵測============
def out_lane_line(event):
    lane_types = set(x.type for x in event.crossed_lane_markings)
    lane_name = [str(lane).split()[-1] for lane in lane_types]
    print(event)


bp = blueprint_library.find('sensor.other.lane_invasion')
lane_line_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
lane_line_sensor.listen(out_lane_line)
actor_list.append(lane_line_sensor)


# ============相機============
def process_rgb_image(img):
    img.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (img.height, img.width, 4))
    frame = array[:, :, :3]
    cv2.imshow('ori', frame)
    cv2.waitKey(1)


camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(process_rgb_image)
actor_list.append(camera)

# ============碰撞偵測============
def collision_info(event):
    print(event)

collision_bp = blueprint_library.find("sensor.other.collision")
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
collision_sensor.listen(collision_info)
actor_list.append(collision_sensor)


while True:
    if(input() is not None):
        break


for actor in actor_list:
    actor.destroy()


# client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
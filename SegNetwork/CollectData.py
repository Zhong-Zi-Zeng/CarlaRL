from agents.navigation.behavior_agent import BehaviorAgent
import carla
import random
import cv2
import numpy as np

# 添加物件都會存於模擬環境中，當程式結束時需要刪除所有物件
# 不然下次再新增時可能會報錯誤
actor_list = []

# 與Sever端建立連接
client = carla.Client('localhost',2000)
client.set_timeout(30.0)

# 建立world物件，模擬環境裡的物件都是由這個來管理
world = client.get_world()
world.tick()
blueprint_library = world.get_blueprint_library()

map = world.get_map()

# ============車子============
bp = random.choice(blueprint_library.filter('vehicle'))
transform = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(bp, transform)
vehicle.set_autopilot(False)
actor_list.append(vehicle)

# ============建立Behavior Agent============
agent = BehaviorAgent(vehicle)
des = random.choice(world.get_map().get_spawn_points())
agent.set_destination(des.location)

# ============回調函式============
count = 0

def callback(img):
    global count

    img = process_bgr_frame(img)
    cv2.imshow('',img)
    cv2.waitKey(1)

    way = agent.getIncomingWaypoint()
    if way is not None:
        angel = countDegree(vehicle, way)
        print(angel)

    control = agent.run_step()
    vehicle.apply_control(control)

    # if(count % 20 == 0):
        # print(count)

        # with open('label.txt','a') as file:
        #     waypoint = getLaneWaypoint()
        #     junction = '1' if waypoint.is_junction else '0'
        #     tl = '1' if str(vehicle.get_traffic_light_state()) == 'Green' else '0'
        #     print('Write to txt file:',count)
        #     file.writelines(str(count) + ' ' + junction + ' ' + tl + '\n')
        #
        # img.save_to_disk('./data/%d.png'%(count))

    # count += 1

# ============建立相機============
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '400')
camera_bp.set_attribute('image_size_y', '300')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(callback)


# ============影像轉ndarray============
def process_bgr_frame(bgr_frame):
    bgr_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(bgr_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (bgr_frame.height, bgr_frame.width, 4))
    bgr_frame = array[:, :, :3]

    return bgr_frame


# ============計算與waypoint的角度============
def countDegree(vehicle, waypoint):
    car_transform = vehicle.get_transform()
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


try:
    while True:
        if(input() is not None):
            break

finally:
    for actor in actor_list:
        actor.destroy()
    print('destroy actor')


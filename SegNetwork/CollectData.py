import carla
import random
import numpy as np
import cv2


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
vehicle.set_autopilot(True)
actor_list.append(vehicle)


# ============相機============
count = 2220

def process_rgb_image(img):
    global count

    if(count % 20 == 0):
        print(count)

        with open('label.txt','a') as file:
            waypoint = getLaneWaypoint()
            junction = '1' if waypoint.is_junction else '0'
            tl = '1' if str(vehicle.get_traffic_light_state()) == 'Green' else '0'
            print('Write to txt file:',count)
            file.writelines(str(count) + ' ' + junction + ' ' + tl + '\n')

        img.save_to_disk('./data/%d.png'%(count))

    count += 1

# ============道路座標點============
def getLaneWaypoint():
    waypoint = map.get_waypoint(vehicle.get_transform().location)
    waypoint = random.choice(waypoint.next(15))

    return waypoint

# ============橫向位移差訊息============


# 一般相機
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '400')
camera_bp.set_attribute('image_size_y', '300')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(process_rgb_image)

try:
    while True:
        if(input() is not None):
            break

finally:
    for actor in actor_list:
        actor.destroy()
    print('destroy actor')


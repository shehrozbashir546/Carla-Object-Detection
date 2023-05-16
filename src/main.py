# code from https://www.youtube.com/watch?v=m1vctOrYrTY&t=1566s

import sys
import math
import glob
import os
import numpy as np
import cv2
import random 
import time
from PIL import Image
import yaml
from yaml.loader import SafeLoader
from refactor import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 640
IM_HEIGHT = 480

def generate_colors(ID):
    np.random.seed(42)
    colors = np.random.randint(100,255,size=(ID,3)).tolist()
    return tuple(colors[ID])

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6*math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def main():
    actor_list = []
    try:
        client = carla.Client('localhost',2000)
        client.set_timeout(2000.0)
        world = client.get_world()
        

        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('model3')[0]
        print(bp)
        
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        #vehicle.set_autopilot(True)

        # adding the camera

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x',f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y',f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov','110')
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
        actor_list.append(sensor)
        print('created %s' % sensor.type_id)
        
        print("Attempting to open the OpenCV Interface")
        sensor.listen(lambda image: processImage(image))
        
        print("OpenCV Interface has been opened successfully")

        time.sleep(120)


    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')


if __name__ == '__main__':
    main()
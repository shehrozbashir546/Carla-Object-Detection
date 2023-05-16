#testing some scripts on the hshl machine

import carla
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)  # Set the timeout for client connection

# Assuming you have a running CARLA simulator, you can retrieve the world
world = client.get_world()


# Retrieve the blueprint for a camera sensor
camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')

# Set the camera attributes (e.g., image size, field of view)
camera_blueprint.set_attribute('image_size_x', '640')
camera_blueprint.set_attribute('image_size_y', '480')
camera_blueprint.set_attribute('fov', '90')

# Spawn the camera sensor at a specific location on the map
camera_spawn_point = carla.Transform(carla.Location(x=2.5, y=0.0, z=1.4))
camera = world.spawn_actor(camera_blueprint, camera_spawn_point)

# Register a callback function to receive the camera image data
def process_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image.raw_data)
    # Reshape the array to match the dimensions of the image
    image_array = image_array.reshape((image.height, image.width, 4))
    # Remove the alpha channel
    image_array = image_array[:, :, :3]
    
    # Perform any desired image processing or analysis here
    
    # Display the image (optional)
    cv2.imshow("Camera Stream", image_array)
    cv2.waitKey(1)  # Display the image for 1 millisecond

# Bind the callback function to the camera sensor
camera.listen(process_image)



# Start the simulation by ticking the CARLA world
world.tick()

# Continue the simulation indefinitely (you may add a condition to break the loop)
while True:
    world.tick()


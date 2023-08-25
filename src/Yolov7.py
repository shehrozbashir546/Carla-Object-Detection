import carla 
import math 
import random 
import time 
import numpy as np
import cv2
import yaml
import random 
from yaml.loader import SafeLoader
import onnx
import onnxruntime as ort


# change the second line to the path of your weights 
# this is what starts the onnx runtime 
session = ort.InferenceSession(
        'yolo7.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
output_type = session.get_outputs()[0].type
# this is just to see what everything looks like, can be removed 
print("input name", input_name)
print("input shape", input_shape)
print("input type", input_type)
print("output name", output_name)
print("output shape", output_shape)
print("output type", output_type)

# same as the previous code block but a different shorter way to achieve it
outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]


# Connect the client and set up bp library and spawn points
# standard carla set up 
client = carla.Client('localhost', 2000) 
world = client.get_world()
bp_lib = world.get_blueprint_library() 
spawn_points = world.get_map().get_spawn_points() 
settings = world.get_settings()
settings.fixed_delta_seconds = 0.01
world.apply_settings(settings)
# path to your yaml file
with open('data.yaml', mode='r') as f:
    data_yaml = yaml.load(f,Loader=SafeLoader)

CLASSES = data_yaml['names']
print('The Classes are as follows:',CLASSES)


# Spawn ego vehicle, rgb camera and pedestrians 
vehicle_bp = bp_lib.find('vehicle.audi.a2') 
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
# Move spectator behind vehicle to view
spectator = world.get_spectator() 
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=1,z=2)),vehicle.get_transform().rotation) 
spectator.set_transform(transform)
camera_bp = bp_lib.find('sensor.camera.rgb') 
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '640')
# this location is important, if you set it too close
# the object detection starts predicting the ego vehicle as
# a car, which is not what i want

camera_init_trans = carla.Transform(carla.Location(x=1,z=2)) #Change this to move camera
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
# time.sleep cuz carla is a bit slow slow 
# allows carla to catch up with the camera being initialized
time.sleep(0.2)
spectator.set_transform(camera.get_transform())
# Get camera dimensions and initialise dictionary                       
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

camera_data = {'image': np.zeros((image_h, image_w, 4))}
print('Camera width is ',image_w,' and camera height is ',image_h)

# spawn 50 vehicles 

for i in range(50): 
    vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 
for v in world.get_actors().filter('*vehicle*'): 
    v.set_autopilot(True) 

# this is what you use to start the object detection 

camera.listen(lambda image: camera_callback(image, camera_data))
cv2.waitKey(1)

# Game loop
while True:
    

    # Imshow renders sensor data to display
    cv2.imshow('YOLOv7 Detections', camera_data['image'])

    if cv2.waitKey(1) == ord('q'):
        break

# Close OpenCV window when finished
cv2.destroyAllWindows()

# tadaaa, everything works!


# use this after you have closed the opencv predictions window
# CARLA has some crazy memory leaks 
camera.destroy()
for v in world.get_actors().filter('*vehicle*'): 
    v.destroy()


# Callback stores sensor data in a dictionary for use outside callback    
# also used for object detection          
def camera_callback(image, data_dict):
    
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # ONNX expects things in a particular format
    # so the CARLA camera data needs to be reshaped accordingly
    # first remove the alpha channel

    
    frame = cv2.cvtColor(data_dict['image'], cv2.COLOR_RGBA2RGB)
    # then transpose it 
    transposed_image = frame.transpose((2,0,1))
    # expand the dimensions, sounds cool but its not 

    expanded_dimensions = np.expand_dims(transposed_image,0)
    # change to float because it does not like int 
    inputImage = expanded_dimensions.astype(np.float32)
    # normalize because everyone likes nice numbers
    inputImage /= 255
    # start infering with ONNX
    outputs = session.run(outname, {input_name: inputImage})[0]
    # this is the source image needed for opencv to display the preds
    image = data_dict['image']
    # you can use this if you want, i dont like the color randomizer
    # it basically randomizes the bounding box color that will come on the
    # predictions
    colorRandomizer = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # YOLOv7 comes with built in NMS so this is all that is needed to display
    # the predictions
    # super convenient!

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        
        box = np.array([x0,y0,x1,y1])
        box = box.round().astype(np.int32).tolist()
        class_id = int(cls_id)
        floatscore = round(float(score),3)
        name = CLASSES[class_id]
        objectScore = int(floatscore*100)
        color = colorRandomizer[class_id]
        # you can print this text to see everything in the console as well
        text = f'{name}: {objectScore}%'
        # change the color to (0,0,0) or some other rgb value
        # if you dont want the randomizer
        cv2.rectangle(image, box[:2],box[2:] , color ,2 ) 
        cv2.putText(image,text,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0, 0, 0],thickness=2) 
        # slows down the simulation in order to show the predictions, otherwise it goes fast fast 

        time.sleep(0.05)

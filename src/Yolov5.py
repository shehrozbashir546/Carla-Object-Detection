import carla 
import math 
import random 
import time 
import numpy as np
import cv2
import yaml
from yaml.loader import SafeLoader

# change to the path of your weights 
net = cv2.dnn.readNetFromONNX('yolov5weights.onnx')
print(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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

# thresholds, change according to your use case 
# or dont! 

conf_threshold = 0.5
class_threshold = 0.4
# variable declarations 
IM_WIDTH = 640
IM_HEIGHT = 640


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

time.sleep(0.2)
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
    cv2.imshow('YOLOv5 Detections', camera_data['image'])

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
    frame = cv2.cvtColor(data_dict['image'], cv2.COLOR_RGBA2RGB)
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop = False)

    net.setInput(blob)
    # forward pass
    preds = net.forward()
    # create empty arrays for storing
    detections = preds[0] 
    boxes = []
    confidences = []
    classes = []
    
    imageW = camera_bp.get_attribute("image_size_x").as_int()
    imageH = camera_bp.get_attribute("image_size_y").as_int()
    # factor isnt that important if your carla camera is already 640*640
    x_factor = imageW/640
    y_factor = imageH/640

    # this for loop extracts all the predictions based conf and class thresh
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] 
        if confidence > conf_threshold:
            # getting the best score and then get its index position, id stands for index
            class_score = row[5:].max() 
            class_id = row[5:].argmax()

            if class_score>class_threshold:
                cx,cy,w,h = row[0:4]
                # coordinates need to be scaled 
                left = int((cx-0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                # append everything so you can access it later for nms 
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)

    # now convert it to the format nms wants, honestly this step could be skipped
    boxes_array = np.array(boxes)
    boxes_np = boxes_array.tolist()
    confidences_array = np.array(confidences)
    confidences_np = confidences_array.tolist()
    # .flatten() can be removed but your milleage may vary
    # it was quite finnicky in my testing
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
    
    # need the source image to tell opencv what to display it on
    # i4 is the source image coming from the camera sensor
    i4 = data_dict['image']
    # extract the predictions post NMS
    # now you are ready to display everything
    for ind in index:
        x,y,w,h = boxes_np[ind]
        bb_conf = int(confidences_np[ind]*100)
        classes_id = classes[ind]
        class_name = CLASSES[classes_id]

        text = f'{class_name}: {bb_conf}%'
        # print is just to show it in the console, you can remove this if you dont want anything verbose
        print(text)

        cv2.rectangle(i4,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.putText(i4,text,(x,y-5),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
        
        # slows down the simulation in order to show the predictions, otherwise it goes fast fast 
        time.sleep(0.05)

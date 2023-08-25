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
        '/home/ageda/projects/Yolov8/runs/detect/ShehrozThesisYolov8/weights/ShehrozThesisYolov8.onnx',
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
# same as the previous code but a different shorter way to achieve it

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

with open('data_custom.yaml', mode='r') as f:
    data_yaml = yaml.load(f,Loader=SafeLoader)
CLASSES = data_yaml['names']

print('The Classes are as follows:',CLASSES)

# thresholds, change according to your use case 
# or dont! 
conf_threshold = 0.5
class_threshold = 0.4


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
    cv2.imshow('YOLOv8 Detections', camera_data['image'])

    if cv2.waitKey(1) == ord('q'):
        break

# Close OpenCV window when finished
cv2.destroyAllWindows()
# tadaaa, everything works!

# Callback stores sensor data in a dictionary for use outside callback    
# also used for object detection  
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    
    
    # need the source image to tell opencv what to display it on
    # originalImage is the source image coming from the camera sensor
    originalImage = data_dict['image']
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

    output = session.run(outname, {input_name: inputImage})[0]

    #[8.96539593e+00 5.91810465e-01 1.15975008e+01 9.05570602e+00
    #0.00000000e+00 0.00000000e+00 0.00000000e+00 4.76837158e-07
    #0.00000000e+00 0.00000000e+00 1.06977284e-01 4.38958406e-03
     #9.98617768e-01 2.87176967e-02] these are the coordinates of the bounding boxes
     
    outputs = np.transpose(np.squeeze(output))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        # get the best score 
        max_score = np.amax(classes_scores)
        if max_score >= conf_threshold:
                # Get the class index of the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) )
                top = int((y - h / 2) )
                width = int(w)
                height = int(h)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
    #apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.35)
    # extract the predictions post NMS
    # now you are ready to display everything
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        name = CLASSES[class_id]
        x1, y1, w, h = box
        DisplayedScore = int(score*100)
        text = f'{name}: {DisplayedScore}%'
        # you can comment this out if you dont want the text displayed
        print(text)
        #print('coordinates',x1,y1,'for',text)
        cv2.rectangle(originalImage, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),(0,0,0) , 2)
        
        # -5 seemed like a good height to display the text
        # adjust it if you dont like it
        cv2.putText(originalImage,text,(int(x1),int(y1-5)),cv2.FONT_HERSHEY_PLAIN,0.7,(255,255,255),1)
        # slows down the simulation in order to show the predictions, otherwise it goes fast fast 

        time.sleep(0.05)


# use this after you have closed the opencv predictions window
# CARLA has some crazy memory leaks 
camera.destroy()
for v in world.get_actors().filter('*vehicle*'): 
    v.destroy()

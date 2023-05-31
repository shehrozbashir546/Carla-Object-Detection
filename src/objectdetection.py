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
#from refactor import *
from PIL import Image as im
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
 
import carla
 
# Load the model
# get the labels
# boiler plate code carla
#   spawn camera in front of car
#   turn on autopilot
# sensor listen
# process the image
# make a detector function that takes the model
# forward the blobs
# extract labels
# display the labels
# use the im function to turn an array into an image
 
net = cv2.dnn.readNetFromONNX('weights/perfect.onnx')
#print(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IM_WIDTH = 640
IM_HEIGHT = 480
conf_threshold = 0.4
class_threshold = 0.25
with open('data/data.yaml', mode='r') as f:
    data_yaml = yaml.load(f,Loader=SafeLoader)
labels = data_yaml['names']
nc = data_yaml['nc']
#print('does this even run')
 
def main():
    actor_list = []
    try:
        client = carla.Client('localhost',2000)
        client.set_timeout(2000.0)
        world = client.get_world()
        
 
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('model3')[0]
        spawn_points = random.choice(world.get_map().get_spawn_points())
        
        #transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_points)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
 
        # adding the camera
 
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x',f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y',f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov','110')
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
        actor_list.append(sensor)
        print('created %s' % sensor.type_id)
 
        # Move the spectator behind the vehicle to view it
        spectator = world.get_spectator() 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
        spectator.set_transform(transform)
 
        sensor.listen(processImage)
        
        while True:
            world.tick()
            time.sleep(10)
 
 
    except Exception as e:
        print(e)
 
    finally:
 
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')
 
 
def processImage(image):
    print('receiving image...')
    #print(image.raw_data)
    i = np.array(image.raw_data)
    #print(i, i.shape)
    #i1 = i.copy()
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4))
    i3 = i2[:,:,:3]
    print('i3 is /n',i3)
    img_RGB = cv2.cvtColor(i2, cv2.COLOR_RGBA2RGB)
    #img_RGB2 = cv2.cvtColor(image.raw_data, cv2.COLOR_RGBA2RGB)
    print('cvtColor is /n',img_RGB)
    #print(img_RGB)
    data = im.fromarray(i3)
    #i3= image.raw_data
    #img = Image.fromarray(np.uint8(cv2.cvtColor(i,cv2.COLOR_BGR2RGB)))
    #detections = yolo(data)
    #blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
#    cv2.imshow('OpenCV Interface of detections', img_RGB)
    # #cv2.imshow('OpenCV Interface of img',i3)
    # #cv2.waitKey(1)
    #k = cv2.waitKey(100) & 0xFF
    #if k == 27:         # wait for ESC key to exit
    #    cv2.destroyAllWindows()
    return i3/255.0
 
def yolo(i3):
    frame = cv2.cvtColor(i3, cv2.COLOR_RGBA2RGB)
    layers_names = net.getLayerNames()
    outputLayers=[]
    for i in net.getUnconnectedOutLayers():
        outputLayers.append(layers_names[i-1])
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop = False)
    print("blob boi WORKS YEAHHHH\n")
    net.setInput(blob)
    preds = net.forward() # this gets your detections and predictions from yolo
    #print(preds.shape)
    print("PREDS JUST WORKED!!!!!")
    detections = preds[0] 
    boxes = []
    confidences = []
    classes = []
    
    img = i3
    
    imageW, imageH = img.shape[:2]
    #print('Image W is ',imageW)
    #print('ImageH is ',imageH)
    x_factor = imageW/IM_WIDTH
    y_factor = imageH/IM_HEIGHT
    
    for i in range(len(detections)): #runs 25200 times
            row = detections[i]
            confidence = row[4] 
            if confidence > conf_threshold:
                class_score = row[5:].max() 
                class_id = row[5:].argmax()
                
                if class_score>class_threshold:
                    cx,cy,w,h = row[0:4]
                    left = int((cx-0.5*w)*x_factor)
                    top = int((cy-0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    box = np.array([left,top,width,height])
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)
 
 
    #print("Detections have been filtered based on confidence")
    #print("now we print the boxes", boxes)
    #print("now we print the class ids",classes)
    boxes_array = np.array(boxes)
    boxes_np = boxes_array.tolist()
    confidences_array = np.array(confidences)
    confidences_np = confidences_array.tolist()
    #print('boxes_np =',boxes_np)
    #print('confidences_np = ',confidences_np)
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
    print("the index positions are: ",index)
    #print("Now I attempt to draw the boxes")
    i4 = cv2.resize(i3, (IM_WIDTH, IM_HEIGHT), None)
    for ind in index:
        
        x,y,w,h = boxes_np[ind]
        bb_conf = int(confidences_np[ind]*100)
        classes_id = classes[ind]
        class_name = labels[classes_id]
        
        #colors = generate_colors(classes_id)
 
        text = f'{class_name}: {bb_conf}%'
        print(text)
        #2:33
        cv2.rectangle(i4,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.rectangle(img,(x,y),(x+w,y+h),colors,2)
        cv2.rectangle(i4,(x,y-15),(x+w,y),(255,255,255),-1)
        cv2.putText(i4,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
    return i3
 
if __name__ == '__main__':
    main()
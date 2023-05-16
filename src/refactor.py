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

IM_WIDTH = 640
IM_HEIGHT = 480

def processImage(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4))
    i3 = i2[:,:,:3]

    #detected_image = detector(i3)
    #cv2.imshow('OpenCV Interface of detections',detected_image)
    cv2.imshow('OpenCV Interface of img',i3)
    cv2.waitKey(1)
    return i3/255.0


# this function loads the model and 
def ModelLoader():
    yolo = cv2.dnn.readNetFromONNX('best.onnx')
    print("Yolo ONNX Model has been loaded")
    # we have to specify which gpu or cpu we are using
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    #gpu
    #yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    return yolo

def detector2(i3):
    test_im = i3
    frameWidth= 640
    frameHeight = 480
    net = cv2.dnn.readNetFromONNX('onnxtest2.onnx')

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()] # we put the names in to an array

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] -1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size = (len(classes), 3))
    # Loading image
    #img = cv2.imread(args.image)
    img = cv2.resize(test_im, (frameWidth, frameHeight), None)
    height, width, channels = img.shape
    # Detect image
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop = False)

    net.setInput(blob)
    print("am i still having issues in the forward function")
    outs = net.forward(output_layers)
    print(" I AM NOT")
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y -h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                # Name of the object
                class_ids.append(class_id)
    #print(center_x, center_y)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes),3))
    for i in range(len(boxes)):
        if i in indexes:
    #for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x + 60, y + 60), font, 0.8, (255, 255, 200), 1)



def blobTest(i3, yolo):
    
    ##Now the issue is the .forward  mf

    #img = Image.fromarray(np.uint8(cv2.cvtColor(i3,cv2.COLOR_BGR2RGB)))
    #print(img)
    #blob = cv2.dnn.blobFromImage(img,1/255,(inputW,inputH),swapRB=True,crop=False)
    
    layers_names = yolo.getLayerNames()
    UnconnectedOutLayers = yolo.getUnconnectedOutLayersNames()
    #output_layers = [layers_names[i[0] -1] for i in UnconnectedOutLayers]
    #here are my output layers
    #print("this is what the output layers look like")
    #print(output_layers)
    frame = cv2.cvtColor(i3, cv2.COLOR_RGBA2RGB)  
    outputLayers=[]
    for i in yolo.getUnconnectedOutLayers():
        outputLayers.append(layers_names[i-1])
    #img = cv2.resize(i3, (640, 480), None)
    print("Image resized and output layers worked!")
    #print(img)
    #height, width, channels = img.shape
    # Detect image
    
    #
    #print("time to set what the  blob looks like")
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop = False)
    print("blob boi WORKS YEAHHHH\n")
    yolo.setInput(blob)
    #print("Did my preds variable work?")
    preds = yolo.forward() # this gets your detections and predictions from yolo
    #print(preds.shape)
    print("PREDS JUST WORKED!!!!!")
    return preds
#Explaining YOLO output and the preds
#Now the shape of the predictions is 1, 25200 , 10
#this means 25200 rows are the number of bounding boxes from the image
#the 10 columns are split into two parts:
# the first 5 columns are Center X, Center Y, W, H and the Confidence Scores
# Center X and Center Y are the center point of the bounding box which is normalized to width and height respectively
# W and H is the width and height of the bounding box which is normalized to the w and h of the image respectively
# Confidence score is the confidence of detecting the bounding boxes
# the remaining columns are the probability scores((classification scores) of each class
# Since we have 5 classes we have 5 columns. More classes means more columns
# there are duplicated detections so NMS needs to be applied to remove them
# 
def detector(i3):
    with open('data.yaml', mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)
    labels = data_yaml['names']
    
    #NC is the number of labels
    nc = data_yaml['nc']
    print("Our labels are: {} and the quantity is {}".format(labels,nc))
    #print("Yaml File has been loaded")
    inputW = 640
    inputH = 480
    #loading the yolo model
    yolo = ModelLoader()

    preds = blobTest(i3,yolo)
    print("Detections and Predictions have been extracted from yolo")

    # non maximum suppression? no idea what this is  ( we do non max to get rid of duplicates)
    #step 1: filter detections based on confidence score
    # conf threshold is 0.4 and prob score is thres is 0.25
    detections = preds[0] # doing this flattens the preds to just 25200 columns and 10 rows
    boxes = []
    confidences = []
    classes = []
    #width and height of the image 
    img = i3
    #print('im printing the shape of img',img.shape)
    imageW, imageH = img.shape[:2]
    #print('Image W is ',imageW)
    #print('ImageH is ',imageH)
    x_factor = imageW/inputW
    y_factor = imageH/inputH
    #print(x_factor,y_factor)
    conf_threshold = 0.4
    class_threshold = 0.25
    #filter out the detections
    for i in range(len(detections)): #runs 25200 times
        row = detections[i]
        # 4th column is the confidence 
        confidence = row[4] # confidence of object detects

        # only consider confidences greater than the threshold
        if confidence > conf_threshold:
            # 5th column is the start of prob scores
            class_score = row[5:].max() # max probs of detected objs

            #get the index position of the max probs
            class_id = row[5:].argmax()
            
            if class_score>class_threshold:
                # same values mentioned in the paragraph above
                cx,cy,w,h = row[0:4]
                
                # making the bboxes from these 4 values
                # left, top  width and height
                left = int((cx-0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                # save all this info into an array
                box = np.array([left,top,width,height])
                
                #append everything to our original lists
                
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)
                
                # now all our information is extracted

    print("Detections have been filtered based on confidence")

    #print(confidences)
    #print("now we print the boxes", boxes)
    #print("now we print the class ids",classes)
    # turn into an np array
    boxes_array = np.array(boxes)
    # turn into a list
    boxes_np = boxes_array.tolist()

    confidences_array = np.array(confidences)
    confidences_np = confidences_array.tolist()

    #print('boxes_np =',boxes_np)
    #print('confidences_np = ',confidences_np)
    #non max suppress time to get rid of dupe values
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
    #print("the index positions are: ",index)
    #NMS boxes requires bboxes, scores, scorethreshold,nmsthreshold
    # drawing bounded boxes
    print("Now I attempt to draw the boxes")

    #testing with frame instead of img
    frame = cv2.cvtColor(i3, cv2.COLOR_RGBA2RGB)
    #placeholder = img
    placeholder = frame
    for ind in index:
        #extracting the bboxes
        x,y,w,h = boxes_np[ind]
        bb_conf = int(confidences_np[ind]*100)
        classes_id = classes[ind]
        class_name = labels[classes_id]
        
        #colors = generate_colors(classes_id)

        text = f'{class_name}: {bb_conf}%'
        print(text)
        #2:33
        cv2.rectangle(placeholder,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.rectangle(img,(x,y),(x+w,y+h),colors,2)
        cv2.rectangle(placeholder,(x,y-15),(x+w,y),(255,255,255),-1)
        cv2.putText(placeholder,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)

    return img

#this function can be used to generate colors for the bbox labels
#based on their class IDs 
#just replace the numerical values with colors
#i can also just take a size of 5 because there are 5 classes
def generate_colors(ID):
    np.random.seed(42)
    colors = np.random.randint(100,255,size=(ID,3)).tolist()
    return tuple(colors[ID])

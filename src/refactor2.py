import numpy as np
import cv2
from PIL import Image


IM_WIDTH = 640
IM_HEIGHT = 480

def processImage(image, preprocessed_yolo):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4))
    i3 = i2[:,:,:3]

    detected_image = detector(i3,preprocessed_yolo)
    cv2.imshow('OpenCV Interface of detections',detected_image)
    #cv2.imshow('OpenCV Interface of img',i3)
    #cv2.waitKey(1)
    return i3/255.0


def detector(i3,preprocessed_yolo,labels):
   
    frame = cv2.cvtColor(i3, cv2.COLOR_RGBA2RGB)  
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop = False)
    print("blob boi WORKS YEAHHHH\n")

    preprocessed_yolo.setInput(blob)
    preds = preprocessed_yolo.forward() 
    # this gets your detections and predictions from yolo
    #NC is the number of labels
    
    #print("Yaml File has been loaded")
    inputW = 640
    inputH = 480
    detections = preds[0] # doing this flattens the preds to just 25200 columns and 10 rows
    boxes = []
    confidences = []
    classes = []
    #width and height of the image 
    img = i3
    #print('im printing the shape of img',img.shape)
    imageW, imageH = img.shape[:2]
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

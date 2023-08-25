# Carla-Object-Detection

Bachelor Thesis - Hochschule Hamm Lippstadt - Bachelors in Electronic Engineering

Running Object Detection Algorithms inside CARLA. 


YOLOv5, YOLOv7 and YOLOv8 are trained and used for real time inference inside CARLA.


Jupyter contains the notebooks used in this implementation

For YOLOv5, OpenCV DNN is used to load the weights

For YOLOv7 and 8, the ONNX Runtime is used. The YOLOv7 weights have already been exported with Non-maximum supression and its code base reflects that. 

YOLOv8 however does not support NMS out of the box yet. If you are interested in using the ONNX Runtime for object detection inference, refer to the code for YOLOv8.

Ubuntu LTS 20.04 is used with CARLA 0.9.14. OpenCV is compiled from source with GPU support enabled. 

# Carla-Object-Detection

Bachelor Thesis - Hochschule Hamm Lippstadt - Bachelors in Electronic Engineering

Running Object Detection Algorithms inside CARLA. 
YOLOv5, YOLOv7 and YOLOv8 are trained and used for real time inference inside CARLA

For YOLOv5, OpenCV DNN is used to load the weights

For YOLOv7 and 8, the ONNX Runtime is used
The YOLOv7 weights have already been exported with Non-maximum supression and its code base reflects that

Ubuntu LTS 20.04 is used
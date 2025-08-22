import cv2 as cv
import numpy as numpy
import math 
import matplotlib.pyplot as plt
import json
import os
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

img_dir = "/home/Anjali/Desktop/VisDrone Human set Test/VisDrone Human set Test/images"
ann_dir = "/home/Anjali/Desktop/VisDrone Human set Test/VisDrone Human set Test/labels"


for count in range(1,len(os.listdir(img_dir)), 120):
    img = f"VisDrone_human_{count}.jpg"
    img_path = os.path.join(img_dir, img)
    image = cv.imread(img_path)
    img_pred = cv.resize(image,(640,640))
    results = model.predict(img_pred, classes = [0], conf = 0.3)
    print("Reading image: ", img_path)
    height, width, _ = img_pred.shape
    file_name, ext = os.path.splitext(img)
    ann_path = f"{ann_dir}/{file_name}.txt"
    with open(ann_path, 'r') as f:
        annotations = f.readlines()

        for annotation in annotations:
            # Parse each line in the annotation file
            data = annotation.strip().split()
            class_id = int(data[0])  # Class ID (not used for drawing, but can be used for labeling)
            x_center = float(data[1]) * width
            y_center = float(data[2]) * height
            bbox_width = float(data[3]) * width
            bbox_height = float(data[4]) * height
            
            # Convert the center coordinates and width/height into top-left and bottom-right corners
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)
            
            # Draw the bounding box on the image
            cv.rectangle(img_pred, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        # plt.imshow(image)
        plt.imshow(img_pred)
        plt.show()
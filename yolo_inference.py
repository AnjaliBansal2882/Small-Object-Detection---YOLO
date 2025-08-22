import cv2 as cv
import numpy as numpy
import math 
import matplotlib.pyplot as plt
import json
import os
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

img_dir = "/home/shubhi/Desktop/Anjali_dev/VisDrone Human set Test/VisDrone Human set Test/images"
ann_dir = "/home/shubhi/Desktop/Anjali_dev/VisDrone Human set Test/VisDrone Human set Test/labels"
# for img in os.listdir(img_dir):
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
    for result in results:
        boxes = result.boxes.xywh.tolist()
        print(boxes)
        for box in boxes: 
            center_x, center_y, w, h = box
            top_left_x = center_x - (w / 2)
            top_left_y = center_y - (h / 2)
            bottom_right_x = center_x + (w / 2)
            bottom_right_y = center_y + (h / 2)
            print(center_x, center_y, (center_x+w), (center_y+h),)
            top_left = (int(top_left_x),int(top_left_y))
            bottom_right = (int(bottom_right_x), int(bottom_right_y))
            cv.rectangle(img_pred, top_left, bottom_right, (0, 0, 255), 1)


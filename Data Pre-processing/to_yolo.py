# To get all images in the dataset containing annotations for human/person in its ground truth and convert annotations to YOLO format
'''
The code does the following:
1. goes through each annotation
2. reads image and its width and height
3. starts iterating through each line of the concerned annotation file
4. if the class column matches with huamn or person converts the coordinates of that annotation in YOLO format in the specified folder 
5. sets the flag as 1 
6. the flag is checked at last, if for any annotation file, the flag has been turned to 1 even once, that means the image must contain a human annotation and so that image is copied to the human set image folder
7. new folders are created for storing human containing images and annotations if they don't already exist
'''
import os
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO
import shutil


ann_dir = "/home/anjali/Documents/VisDrone_Dataset/VisDrone2019-DET-test-dev/annotations"
img_dir = "/home/anjali/Documents/VisDrone_Dataset/VisDrone2019-DET-test-dev/images"

yolo_ann_dir = "/home/anjali/Documents/VisDrone_Dataset/VisDrone_Human_Yolo_test/labels"
yolo_img_dir = "/home/anjali/Documents/VisDrone_Dataset/VisDrone_Human_Yolo_test/images"

if not os.path.exists(yolo_img_dir):
    os.makedirs(yolo_img_dir)
    print("made directory for yolo images ")

if not os.path.exists(yolo_ann_dir):
    os.makedirs(yolo_ann_dir)
    print("made directory for yolo annotations ")

count = 1

for file in os.listdir(ann_dir):
    flag = 0
    file_path = os.path.join(ann_dir, file)
    file_name, ext = os.path.splitext(file)
    img_path = f"{img_dir}/{file_name}.jpg"
    img = cv.imread(img_path)
    image_height, image_width, _ = img.shape
    print(file_path," \n", img_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print("Reading line: ", line)
            parts = line.strip().split(",")
            print(parts)
            print(parts[5])
            if (int(parts[5]) == 1) or (int(parts[5]) == 2):              # checking of the image contains any person(==1) or human(==2)
                if (math.floor(((image_height*image_width)/409600)*100) < ((int(parts[2])) * (int(parts[3])))):
                    if (int(parts[7]) == 0) or (int(parts[7]) == 1):
                        flag = 1
                        bbox_left, bbox_top, bbox_width, bbox_height = map(float, parts[0:4])
                        x_center = bbox_left + (bbox_width / 2)
                        y_center = bbox_top + (bbox_height / 2)
                        
                        x_center_normalized = x_center / image_width
                        y_center_normalized = y_center / image_height
                        bbox_width_normalized = bbox_width / image_width
                        bbox_height_normalized = bbox_height / image_height

                        annotation_line = f"0 {x_center_normalized:.4f} {y_center_normalized:.4f} {bbox_width_normalized:.4f} {bbox_height_normalized:.4f}\n"

                        with open(f"{yolo_ann_dir}/VisDrone_human_{count}.txt", 'a') as f:
                            print(f"writing {annotation_line} for img: ",file_name)
                            f.write(annotation_line)
    
            else:
                continue
    
    if flag ==1:
        shutil.copy(img_path, f"{yolo_img_dir}/VisDrone_human_{count}.jpg")
        print("Dest path: ", f"{yolo_img_dir}/VisDrone_human_{count}.jpg")
        count+= 1


print("no of human in test set images: ", count)

        
    
    



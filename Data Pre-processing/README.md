This folder contains the conversion scripts and some pre processing functions often required to be applied on a dataset to convert it into model-accaptable format. 

- Here we convert the custom ground truth files of the dataset into the **YOLO** and **COCO** format, the two most common annotation formats in object detection. 
- We also filter for a specific object in dataset
- Filter outs the objects covering less than area less than a threshold % of the image

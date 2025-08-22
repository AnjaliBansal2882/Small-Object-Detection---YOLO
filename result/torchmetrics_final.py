from ultralytics import YOLO
import torch
import os
from torchmetrics.detection import MeanAveragePrecision
import cv2
import json

models = ['yolov8n.pt','runs/detect/origi_new_yolov8n_FT/weights/best.pt', 'runs/detect/origi_new_yolov8n_lp/weights/best.pt', 'runs/detect/origi_new_yolov8n_LPFT/weights/best.pt', 'runs/detect/origi_new_yolov8n_scratch/weights/best.pt',
          'yolov9t.pt','runs/detect/origi_new_yolov9t_FT/weights/best.pt', 'runs/detect/origi_new_yolov9t_lp/weights/best.pt', 'runs/detect/origi_new_yolov9t_LPFT/weights/best.pt', 'runs/detect/origi_new_yolov9t_scratch/weights/best.pt', 
          'yolov10n.pt','runs/detect/origi_new_yolov10n_FT/weights/best.pt', 'runs/detect/origi_new_yolov10n_lp/weights/best.pt', 'runs/detect/origi_new_yolov10n_LPFT/weights/best.pt', 'runs/detect/origi_new_yolov10n_scratch/weights/best.pt',
          'yolo11n.pt','runs/detect/origi_new_yolo11n_FT/weights/best.pt', 'runs/detect/origi_new_yolo11n_lp/weights/best.pt', 'runs/detect/origi_new_yolo11n_LPFT/weights/best.pt', 'runs/detect/origi_new_yolo11n_scratch/weights/best.pt',
          'yolo12n.pt','runs/detect/origi_new_yolo12n_FT/weights/best.pt', 'runs/detect/origi_new_yolo12n_lp/weights/best.pt', 'runs/detect/origi_new_yolo12n_LPFT/weights/best.pt', 'runs/detect/origi_new_yolo12n_scratch/weights/best.pt']

folder = '/home/Anjali/Documents/VisDrone runs/VisDrone2019-DET-test-dev-copy/images'
files = os.listdir(folder)
ious = [0.5,0.6,0.7,0.8,0.9]
def parse_until_slash(input_string,num):
    dot_count = 0
    for i, char in enumerate(input_string):
        if char == '/':
            dot_count += 1
        if dot_count == num:
            return i
    
    return input_string 

for model in models:
    if len(model)<12:
        name=f'case2_{model[:-3]}_pretrained'
    else:
        start = parse_until_slash(model,2)+1
        end = parse_until_slash(model,3)
        name = f'case2_{model[start:end]}'
    data = {
        "mAP0.5":0, "mAP0.6":0, "mAP0.7":0, "mAP0.8":0, "mAP0.9":0,
        "mAR0.5":0, "mAR0.6":0, "mAR0.7":0, "mAR0.8":0, "mAR0.9":0
    }
    model = YOLO(model)
    for iou in ious:
        metric = MeanAveragePrecision(iou_thresholds=[iou], class_metrics=True, max_detection_thresholds = [10,100,500])
        #model = YOLO('/home/shubhi/Downloads/yolov8n.pt') 
        #model.val
        for j in range(len(files)):
            print(j)
            #print(files[i])
            out = model.predict(folder+'/'+files[j],classes = [0], imgsz = 640)
            a = out[0].to_df()

            bboxes = torch.tensor([])
            labels = torch.tensor([])
            scores = torch.tensor([])

            if a.shape != (0,0):
                num = a['box'].shape[0]

                for i in range(num):
                    bbox = a['box'][i]
                    score = torch.tensor([a['confidence'][i]])
                    label = torch.tensor([int(a['class'][i])])
                    #print(type(a['class'][i]))
                    #print(type(int(a['class'][i])))
                    bbox = torch.tensor([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]])
                    bboxes = torch.cat((bboxes,bbox),dim = 0)
                    labels = torch.cat((labels,label),dim=0)
                    scores = torch.cat((scores,score), dim = 0)
                    #print(i)
                    #print(bboxes,labels,scores)

            predictions = [
                {
                    'boxes': bboxes,  
                    'labels': labels.to(torch.int64),
                    'scores': scores
                }
            ]
            #print(i)
            #print(folder)
            #print(files)
            label_file = folder[:-6]+'labels/'+files[j][:-3]+'txt'

            gt_bboxes = torch.tensor([])
            gt_labels = torch.tensor([])

            with open(label_file, encoding="utf-8") as file:
                img = cv2.imread(folder+'/'+files[j])
                for row in [x.split(" ") for x in file.read().strip().splitlines()]:
                    xc = float(row[1])*img.shape[1]
                    yc = float(row[2])*img.shape[0]
                    w = float(row[3])*img.shape[1]
                    h = float(row[4])*img.shape[0]
                    bbox = torch.tensor([[float(xc-w/2), float(yc-h/2), float(xc+w/2), float(yc+h/2)]])
                    label = torch.tensor([int(row[0])])
                    gt_bboxes = torch.cat((gt_bboxes,bbox),dim = 0)
                    gt_labels = torch.cat((gt_labels,label),dim=0)

            ground_truth = [
                {
                    'boxes': gt_bboxes, 'labels': gt_labels.to(torch.int64)
                }
            ]
            
            metric.update(predictions, ground_truth)

        mAP = metric.compute()
        print(mAP)
        data['mAP'+str(iou)]=float(mAP['map'])
        data['mAR'+str(iou)]=float(mAP['mar_500'])
    with open('metrics/'+name+"_torchmetrics.txt", 'w') as json_file:
        json.dump(data, json_file, indent=4)  

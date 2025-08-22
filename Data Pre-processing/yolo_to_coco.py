import os
import json
from PIL import Image

def convert_yolo_to_coco(img_dir, label_dir, class_list, output_json_path):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for i, cls in enumerate(class_list):
        coco_output["categories"].append({
            "id": i,
            "name": cls,
            "supercategory": "none"
        })

    annotation_id = 1
    image_id = 1

    for filename in os.listdir(img_dir):
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        width, height = Image.open(img_path).size

        coco_output["images"].append({
            "file_name": filename,
            "id": image_id,
            "width": width,
            "height": height
        })

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    # cls_id, x, y, w, h = map(float, line.strip().split())

                    x, y, w, h = map(float, line.strip().split()[1:])
                    cls_id = int(line.strip().split()[0])
                    # Convert YOLO to COCO bbox (x, y, width, height)
                    x_min = (x - w / 2) * width
                    y_min = (y - h / 2) * height
                    bbox_width = w * width
                    bbox_height = h * height

                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(cls_id),
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        image_id += 1

    # Save to JSON
    with open(output_json_path, "w") as out_file:
        json.dump(coco_output, out_file, indent=4)


    print(f"COCO annotation file saved to {output_json_path}")

convert_yolo_to_coco(
    img_dir="/home/Anjali/Desktop/VisDrone_Human_Yolo_test_all_categories/images",                  # images/
    label_dir="/home/Anjali/Desktop/VisDrone_Human_Yolo_test_all_categories/labels",           # labels/
    class_list=["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"],       # List your classes here
    output_json_path="coco_final.json"
)


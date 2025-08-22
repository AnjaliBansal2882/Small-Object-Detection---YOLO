from ultralytics import YOLO
import json

models = ['yolov8n.pt','yolov10n.pt','yolo11n.pt']

yaml_file_path = "/home/Anjali/Desktop/data.yaml"
test_yaml_file_path = "/home/Anjali/Desktop/test_data.yaml"    

for mod in models:
    with open("YOLO FT results.txt", 'a') as file:

        model = YOLO(mod)
        model_name = mod[:-3]
        
        results = model.train(data = yaml_file_path, epochs = 200, pateince = 20, batch = 16, imgsz = 640, device = 0, name = f"../{model_name}_ FT")

        # For evaulating the model
        FT_model_path = f"../{model_name}_ FT/train/weights/best.pt"
        model = YOLO( FT_model_path)
        results = model.val(data = test_yaml_file_path, imgsz = 640, batch = 1)                     
        print(f"\n\nResults for {model_name}  FT\n", results.results_dict, "\n\n")
        file.write(f"\n{model_name}  FT")
        json.dump(results.results_dict, file)
        file.write("\n")
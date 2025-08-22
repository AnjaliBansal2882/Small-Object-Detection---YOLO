from ultralytics import YOLO
import json

models = ['yolov8n.pt','yolov10n.pt','yolo11n.pt']

yaml_file_path = "/home/Anjali/Desktop/data.yaml"
test_yaml_file_path = "/home/Anjali/Desktop/test_data.yaml"                            # val is now test data in this yaml file

model = ['yolov8n.yaml','yolov10n.yaml','yolo11n.yaml']
for mod in models:
    with open("YOLO Scratch results.txt", 'a') as file:
        model = YOLO(mod)
        model_name = mod[:-3]
        model.train(data = yaml_file_path, epochs = 500, pateince = 20, batch = 16, imgsz = 640, device = 0, name = f"../{model_name}_Scratch")

        # Evaluating the model
        scratch_model_path = f"../{model_name}_Scratch/train/weights/best.pt"
        results = model.val(data = test_yaml_file_path, imgsz = 640, batch = 1)
        print(f"\n\nResults for {model_name} Scratch\n", results.results_dict, "\n\n")
        file.write(f"\n{model_name} Scratch")
        json.dump(results.results_dict, file)
        file.write("\n")
        file.close()
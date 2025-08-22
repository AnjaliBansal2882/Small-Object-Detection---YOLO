from ultralytics import YOLO
import json

models = ['yolov8n.pt','yolov10n.pt','yolo11n.pt']

yaml_file_path = "/home/Anjali/Desktop/data.yaml"
test_yaml_file_path = "/home/Anjali/Desktop/test_data.yaml"                           # val is now test data in this yaml file

for mod in models:
    with open("YOLO LPFT results.txt", 'a') as file:

        model = YOLO(mod)
        model_name = mod[:-3]
        # Finding the layer to be freezed
        for name, module in model.named_modules():
            latest = ''.join(name.split('.')[:3])
            print(latest)
        
        #Layer freezing
        for name, module in model.named_modules():
            if latest in name:
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
        
        results = model.train(data = yaml_file_path, epochs = 200, pateince = 20, batch = 16, imgsz = 640, device = 0, freeze = int(latest[-2:]), name = f"../{model_name}_LP")

        # For evaulating the model
        lp_model_path = f"../{model_name}_LP/train/weights/best.pt"
        model = YOLO(lp_model_path)
        results = model.val(data = test_yaml_file_path, imgsz = 640, batch = 1)                     
        print(f"\n\nResults for {model_name} LP\n", results.results_dict, "\n\n")
        file.write(f"\n{model_name} LP")
        json.dump(results.results_dict, file)
        file.write("\n")
                

        # LPFT
        model = YOLO(lp_model_path)
        results = model.train(data = yaml_file_path, epochs = 200, pateince = 20, batch = 16, imgsz = 640, device = 0, name = f"../ /{model_name}_LPFT")
        
        # Evaluating thr LPFT model
        lpft_model_path = f"../{model_name}_LPFT/train/weights/best.pt"
        model = YOLO(lpft_model_path)
        results = model.val(data = test_yaml_file_path, imgsz = 640, batch = 1)                     
        print(f"\n\nResults for {model_name} LPFT\n", results.results_dict, "\n\n")
        file.write(f"\n{model_name} LPFT")
        json.dump(results.results_dict, file)
        file.write("\n")
        file.close()


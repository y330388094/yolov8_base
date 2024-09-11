# 模型评估
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model_yaml = r"/home/taoxifa/Ai_project/yolov8/ultralytics/cfg/models/v8/yolov8s.yaml"
    data_yaml = r"my_coco.yaml"
    pre_model = r"/home/taoxifa/Ai_project/yolov8/runs/detect/train/weights/best.pt"

    # model = YOLO(model_yaml, task='detect').load(pre_model)  # build from YAML and transfer weights
    model = YOLO(pre_model, task='detect')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.val(data=data_yaml, imgsz=640, device=[0, 1])

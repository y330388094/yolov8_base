# 选择不同的数据训练自己的模型
# https://blog.csdn.net/m0_54634272/article/details/137435255
# https://blog.csdn.net/qq_42452134/article/details/135181244
# 超详细YOLOv8目标检测全程概述：环境、训练、验证与预测

from ultralytics import YOLO
from dataset_split_2.split_and_copy import split_and_copy_dataset
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # 数据集整理
    split_and_copy_dataset(r"F:\MoonCake_datasets\images", r"F:\MoonCake_datasets\labels", r"F:\MoonCake_datasets\data")

    model = YOLO("yolov8.yaml")  # build a new model,自定义yaml配置文件, 模型配置文件
    model.train(data="data.yaml", epochs=30, device='cpu')  # train the model，使用自定义的数据配置文件

    model.val(data="data.yaml") # 使用自定义的数据集验证模型

    results = model(r"C:\Users\Yezi\Desktop\人工智能实训\HW2\data\images\00909.jpg")  # predict on an image
    plt.imshow(results[0].plot())
    plt.show()
    
    results = model(r"C:\Users\Yezi\Desktop\人工智能实训\HW2\data\images\100318.jpg")  # predict on an image
    plt.imshow(results[0].plot())
    plt.show()
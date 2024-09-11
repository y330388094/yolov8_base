# 模型训练
from ultralytics import YOLO
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load a model
    model_yaml = r"/yolov8/ultralytics/cfg/models/v8/yolov8s.yaml"
    data_yaml = r"my_coco.yaml"
    pre_model = r"/yolov8/backbone/yolov8s.pt"


    # train 1 加载预训练模型进行训练
    model = YOLO(model_yaml, task='detect').load(pre_model)  # build from YAML and transfer weights
    # model = YOLO(pre_model, task='detect')  # load a pretrained model (recommended for training)

     # Train the model
    # data：数据集配置文件的路径，指定用于训练的数据集配置文件
    # epochs：训练过程中整个数据集将被迭代多少次。
    # imgsz：用于设置输入图像尺寸。训练时，只能是整数32的倍数，测试和预测时，可以使用设置[1920,1080]。
    # batch：每个批次中的图像数量。
    # workers：用于设置数据加载过程中的线程数
    # device：模型运行的设置，多gpu时，用列表表示
    # cos_lr：余弦学习率调度器，设置为True可以帮助模型在训练过程中按照余弦函数的形状调整学习率，从而在训练初期使用较高的学习率，有助于快速收敛，而在训练后期逐渐降低学习率，有助于细致调整模型参数
    # close_mosaic：用于确定是否在最后几个训练周期中禁用马赛克数据增强
    # warmup_epochs：预热学习轮数，学习率从低值逐渐增加到初始学习率，以在早期稳定训练
    # 训练结果存储在runs/detect/train下，可以查看评估的结果和最优模型。
    results = model.train(data=data_yaml, epochs=2000, imgsz=640, batch=16, workers=8, device=[0,1],
                          cos_lr=True, close_mosaic=200, warmup_epochs=10)

    

    # train 2  从0开始训练，训练自己的数据集，修改官方的yolov8n.yaml文档，修改参数如分类数，
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model.train(data="data.yaml", epochs=30, device='cpu')  # train the model
    model.val(data="data.yaml")
    results = model(r"C:\Users\Yezi\Desktop\人工智能实训\HW2\data\images\00909.jpg")  # predict on an image
    plt.imshow(results[0].plot())
    plt.show()
    results = model(r"C:\Users\Yezi\Desktop\人工智能实训\HW2\data\images\100318.jpg")  # predict on an image
    plt.imshow(results[0].plot())
    plt.show()

    


    
    # train 3从预训练模型开始训练，首先下载一个官方预训练好的模型，这里下载的是yolov8n
    model=YOLO("yolov8n.pt")
    model.train(data="data.yaml", epochs=30, device='cpu')  # train the model
    model.val(data="data.yaml")
    results = model(r"C:\Users\Yezi\Desktop\人工智能实训\HW2\data\images\00909.jpg")  # predict on an image
    plt.imshow(results[0].plot())
    plt.show()
    results = model(r"C:\Users\Yezi\Desktop\人工智能实训\HW2\data\images\100318.jpg")  # predict on an image
    plt.imshow(results[0].plot())
    plt.show()




    # 模型转换为onnx
    # load model
    model = YOLO('yolov8m.pt')

    # Export model
    success = model.export(format="onnx")
数据集里面的所有图像都拷贝到dataset/images目录下，把json文件拷贝到dataset/json目录里面，把生成的txt标签文件拷贝到dataset/labels目录下。

1、数据标注
2、训练数据转换 ------------ convert_label.py
3、验证数据转换是否正确 ----- data_validate.py
4、分割数据集  -------------- data_split
步骤：
训练集（Training Set）：
    用于训练深度学习模型的数据集。模型通过反向传播和梯度下降等优化算法来学习数据的特征和模式。
验证集（Validation Set）：
    用于调整模型的超参数、选择模型和进行早停（early stopping）等操作。验证集的性能评估不影响模型的权重和参数。
测试集（Test Set）：
    用于最终评估模型的性能。测试集是模型未曾见过的数据，用于评估模型在真实世界中的泛化能力。
一般而言，数据集分割的比例可能是 70%-15%-15% 或 80%-10%-10%。在深度学习中通常会追求更大的数据集，因此可以有更多的数据用于训练。

数据集分割时需要注意以下几点：
样本均衡： 确保每个分组中的类别分布相似，以避免模型过度适应于某些特定类别。
随机性： 使用随机种子（例如，random_state 参数）以确保划分的重复性。
数据预处理一致性： 确保对数据的任何预处理步骤在所有分组上都是一致的，以防止引入不一致性。


5、模型训练
模型训练有两种方式，直接pip安装ultralytics库的和源码安装的训练方法有差异。
单卡
yolo detect train data=my_coco.yaml model=yolov8s.pt epochs=150 imgsz=640 batch=64 workers=0 device=0
多卡训练
yolo detect train data=my_coco.yaml model=./weights/yolov8s.pt epochs=150 imgsz=640 batch=128 workers= \'0,1,2,3\' device=0,1,2

6.部署
pt模型与onnx模型
pt 模型：它通常需要在不同平台上进行PyTorch的兼容性配置，可能需要额外的工作和依赖处理。
onnx 模型：由于ONNX的独立性，更容易在不同平台和硬件上进行部署，无需担心框架依赖性问题。
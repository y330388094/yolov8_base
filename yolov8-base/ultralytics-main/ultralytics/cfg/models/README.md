https://developer.aliyun.com/article/1430611  分析

nc: 80  # 类别数目，nc代表"number of classes"，即模型用于检测的对象类别总数。
scales: # 模型复合缩放常数，例如 'model=yolov8n.yaml' 将调用带有 'n' 缩放的 yolov8.yaml
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n概览：225层, 3157200参数, 3157184梯度, 8.9 GFLOPs
  s: [0.33, 0.50, 1024]  # YOLOv8s概览：225层, 11166560参数, 11166544梯度, 28.8 GFLOPs
  m: [0.67, 0.75, 768]   # YOLOv8m概览：295层, 25902640参数, 25902624梯度, 79.3 GFLOPs
  l: [1.00, 1.00, 512]   # YOLOv8l概览：365层, 43691520参数, 43691504梯度, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x概览：365层, 68229648参数, 68229632梯度, 258.5 GFLOPs
# YOLOv8.0n backbone 骨干层
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2 第0层，-1代表将上层的输入作为本层的输入。第0层的输入是640*640*3的图像。Conv代表卷积层，相应的参数：64代表输出通道数，3代表卷积核大小k，2代表stride步长。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4 第1层，本层和上一层是一样的操作（128代表输出通道数，3代表卷积核大小k，2代表stride步长）
  - [-1, 3, C2f, [128, True]] # 第2层，本层是C2f模块，3代表本层重复3次。128代表输出通道数，True表示Bottleneck有shortcut。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8 第3层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为80*80*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/8。
  - [-1, 6, C2f, [256, True]] # 第4层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。256代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是80*80*256。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16 第5层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/16。
  - [-1, 6, C2f, [512, True]] # 第6层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。512代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是40*40*512。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32 第7层，进行卷积操作（1024代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*1024（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/32。
  - [-1, 3, C2f, [1024, True]] #第8层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是20*20*1024。
  - [-1, 1, SPPF, [1024, 5]]  # 9 第9层，本层是快速空间金字塔池化层（SPPF）。1024代表输出通道数，5代表池化核大小k。结合模块结构图和代码可以看出，最后concat得到的特征图尺寸是20*20*（512*4），经过一次Conv得到20*20*1024。
# YOLOv8.0n head 头部层
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第10层，本层是上采样层。-1代表将上层的输出作为本层的输入。None代表上采样的size（输出尺寸）不指定。2代表scale_factor=2，表示输出的尺寸是输入尺寸的2倍。nearest代表使用的上采样算法为最近邻插值算法。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为40*40*1024。
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4 第11层，本层是concat层，[-1, 6]代表将上层和第6层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*1024，第6层的输出是40*40*512，最终本层的输出尺寸为40*40*1536。
  - [-1, 3, C2f, [512]]  # 12 第12层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。与Backbone中C2f不同的是，此处的C2f的bottleneck模块的shortcut=False。
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第13层，本层也是上采样层（参考第10层）。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为80*80*512。
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3 第14层，本层是concat层，[-1, 4]代表将上层和第4层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是80*80*512，第6层的输出是80*80*256，最终本层的输出尺寸为80*80*768。
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small) 第15层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。256代表输出通道数。经过这层之后，特征图尺寸变为80*80*256，特征图的长宽已经变成输入图像的1/8。
  - [-1, 1, Conv, [256, 3, 2]] # 第16层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 12], 1, Concat, [1]]  # cat head P4 第17层，本层是concat层，[-1, 12]代表将上层和第12层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*256，第12层的输出是40*40*512，最终本层的输出尺寸为40*40*768。
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium) 第18层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。经过这层之后，特征图尺寸变为40*40*512，特征图的长宽已经变成输入图像的1/16。
  - [-1, 1, Conv, [512, 3, 2]] # 第19层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 9], 1, Concat, [1]]  # cat head P5 第20层，本层是concat层，[-1, 9]代表将上层和第9层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是20*20*512，第9层的输出是20*20*1024，最终本层的输出尺寸为20*20*1536。
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large) 第21层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数。经过这层之后，特征图尺寸变为20*20*1024，特征图的长宽已经变成输入图像的1/32。
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5) 第20层，本层是Detect层，[15, 18, 21]代表将第15、18、21层的输出（分别是80*80*256、40*40*512、20*20*1024）作为本层的输入。nc是数据集的类别数。




1. nc
• 含义： nc代表"number of classes"，即模型用于检测的对象类别总数。
• 示例中的值： 80，表示该模型配置用于检测80种不同的对象。由于默认使用COCO数据集，这里nc=80；
2. scales
• 含义： scales用于定义模型的不同尺寸和复杂度，它包含一系列缩放参数。
• 子参数：n, s, m, l, x表示不同的模型尺寸，每个尺寸都有对应的depth（深度）、width（宽度）和max_channels（最大通道数）。
• depth： 表示深度因子，用来控制一些特定模块的数量的，模块数量多网络深度就深；
• width： 表示宽度因子，用来控制整个网络结构的通道数量，通道数量越多，网络就看上去更胖更宽；
• max_channels： 最大通道数，为了动态地调整网络的复杂性。在 YOLO 的早期版本中，网络中的每个层都是固定的，这意味着每个层的通道数也是固定的。但在 YOLOv8 中，为了增加网络的灵活性并使其能够更好地适应不同的任务和数据集，引入了 max_channels 参数。
3. backbone
主干网络是模型的基础，负责从输入图像中提取特征。这些特征是后续网络层进行目标检测的基础。在YOLOv8中，主干网络采用了类似于CSPDarknet的结构。
• 含义： backbone部分定义了模型的基础架构，即用于特征提取的网络结构。
• 关键组成：
• [from, repeats, module, args]表示层的来源、重复次数、模块类型和参数。
• from：表示该模块的输入来源，如果为-1则表示来自于上一个模块中，如果为其他具体的值则表示从特定的模块中得到输入信息；
• repeats: 这个参数用于指定一个模块或层应该重复的次数。例如，如果你想让某个卷积层重复三次，你可以使用 repeats=3。
• module: 这个参数用于指定要添加的模块或层的类型。例如，如果你想添加一个卷积层，你可以使用 conv 作为模块类型。
• args: 这个参数用于传递给模块或层的特定参数。例如，如果你想指定卷积层的滤波器数量，你可以使用 args=[filters]。
• Conv表示卷积层，其参数指定了输出通道数、卷积核大小和步长。
• C2f可能是一个特定于YOLOv8的自定义模块。
• SPPF是空间金字塔池化层，用于在多个尺度上聚合特征。
4. head
• 含义： head部分定义了模型的检测头，即用于最终目标检测的网络结构。
• 关键组成：
• nn.Upsample表示上采样层，用于放大特征图。
• Concat表示连接层，用于合并来自不同层的特征。
• C2f层再次出现，可能用于进一步处理合并后的特征。
• Detect层是最终的检测层，负责输出检测结果。













## Models

Welcome to the [Ultralytics](https://ultralytics.com) Models directory! Here you will find a wide variety of pre-configured model configuration files (`*.yaml`s) that can be used to create custom YOLO models. The models in this directory have been expertly crafted and fine-tuned by the Ultralytics team to provide the best performance for a wide range of object detection and image segmentation tasks.

These model configurations cover a wide range of scenarios, from simple object detection to more complex tasks like instance segmentation and object tracking. They are also designed to run efficiently on a variety of hardware platforms, from CPUs to GPUs. Whether you are a seasoned machine learning practitioner or just getting started with YOLO, this directory provides a great starting point for your custom model development needs.

To get started, simply browse through the models in this directory and find one that best suits your needs. Once you've selected a model, you can use the provided `*.yaml` file to train and deploy your custom YOLO model with ease. See full details at the Ultralytics [Docs](https://docs.ultralytics.com/models), and if you need help or have any questions, feel free to reach out to the Ultralytics team for support. So, don't wait, start creating your custom YOLO model now!

### Usage

Model `*.yaml` files may be used directly in the [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli) with a `yolo` command:

```bash
# Train a YOLOv8n model using the coco8 dataset for 100 epochs
yolo task=detect mode=train model=yolov8n.yaml data=coco8.yaml epochs=100
```

They may also be used directly in a Python environment, and accept the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Initialize a YOLOv8n model from a YAML configuration file
model = YOLO("model.yaml")

# If a pre-trained model is available, use it instead
# model = YOLO("model.pt")

# Display model information
model.info()

# Train the model using the COCO8 dataset for 100 epochs
model.train(data="coco8.yaml", epochs=100)
```

## Pre-trained Model Architectures

Ultralytics supports many model architectures. Visit [Ultralytics Models](https://docs.ultralytics.com/models) to view detailed information and usage. Any of these models can be used by loading their configurations or pretrained checkpoints if available.

## Contribute New Models

Have you trained a new YOLO variant or achieved state-of-the-art performance with specific tuning? We'd love to showcase your work in our Models section! Contributions from the community in the form of new models, architectures, or optimizations are highly valued and can significantly enrich our repository.

By contributing to this section, you're helping us offer a wider array of model choices and configurations to the community. It's a fantastic way to share your knowledge and expertise while making the Ultralytics YOLO ecosystem even more versatile.

To get started, please consult our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for step-by-step instructions on how to submit a Pull Request (PR) 🛠️. Your contributions are eagerly awaited!

Let's join hands to extend the range and capabilities of the Ultralytics YOLO models 🙏!

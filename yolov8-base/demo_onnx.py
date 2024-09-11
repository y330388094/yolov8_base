# 官方位置：https://docs.openvino.ai/2024/notebooks/004-hello-detection-with-output.html
# 推理部署

import argparse
import time
import cv2
import numpy as np
from openvino.runtime import Core  # pip install openvino -i  https://pypi.tuna.tsinghua.edu.cn/simple
import onnxruntime as ort  # 使用onnxruntime推理用上，pip install onnxruntime，默认安装CPU

# COCO默认的80类
CLASSES = ['ambulance', 'mud_tank_truck', 'slag_car']


class OpenvinoInference(object):
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        ie = Core()
        self.model_onnx = ie.read_model(model=self.onnx_path)
        self.compiled_model_onnx = ie.compile_model(model=self.model_onnx, device_name="CPU", config={})
        self.output_layer_onnx = self.compiled_model_onnx.output(0)

    def predict(self, datas):
        predict_data = self.compiled_model_onnx([datas])[self.output_layer_onnx]
        return predict_data


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, imgsz=(640, 640), infer_tool='openvino'):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """
        self.infer_tool = infer_tool
        if self.infer_tool == 'openvino':
            # 构建openvino推理引擎
            self.openvino = OpenvinoInference(onnx_model)
            self.ndtype = np.single
        else:
            # 构建onnxruntime推理引擎
            self.ort_session = ort.InferenceSession(onnx_model,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                                    if ort.get_device() == 'GPU' else ['CPUExecutionProvider'])

            # Numpy dtype: support both FP32 and FP16 onnx model
            self.ndtype = np.half if self.ort_session.get_inputs()[0].type == 'tensor(float16)' else np.single

        self.classes = CLASSES  # 加载模型类别
        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))  # 为每个类别生成调色板

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.

        Returns:
            boxes (List): list of bounding boxes.
        """
        # 前处理Pre-process
        t1 = time.time()
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
        print('预处理时间：{:.3f}s'.format(time.time() - t1))

        # 推理 inference
        t2 = time.time()
        if self.infer_tool == 'openvino':
            preds = self.openvino.predict(im)
        else:
            preds = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: im})[0]
        print('推理时间：{:.2f}s'.format(time.time() - t2))

        # 后处理Post-process
        t3 = time.time()
        boxes = self.postprocess(preds,
                                 im0=im0,
                                 ratio=ratio,
                                 pad_w=pad_w,
                                 pad_h=pad_h,
                                 conf_threshold=conf_threshold,
                                 iou_threshold=iou_threshold,
                                 )
        print('后处理时间：{:.3f}s'.format(time.time() - t3))

        return boxes

    # 前处理，包括：resize, pad, HWC to CHW，BGR to RGB，归一化，增加维度CHW -> BCHW
    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充
        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    # 后处理，包括：阈值过滤与NMS
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.

        Returns:
            boxes (List): list of bounding boxes.
        """
        x = preds  # outputs: predictions (1, 84, 8400)
        # Transpose the first output: (Batch_size, xywh_conf_cls, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls)
        x = np.einsum('bcn->bnc', x)  # (1, 8400, 84)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:], axis=-1), np.argmax(x[..., 4:], axis=-1)]

        # NMS filtering
        # 经过NMS后的值, np.array([[x, y, w, h, conf, cls], ...]), shape=(-1, 4 + 1 + 1)
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # 重新缩放边界框，为画图做准备
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            return x[..., :6]  # boxes
        else:
            return []


    # 绘框
    def draw_and_visualize(self, im, bboxes, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 6], n is number of bboxes.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """
        # Draw rectangles
        for (*box, conf, cls_) in bboxes:
            # draw bbox rectangle
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          self.color_palette[int(cls_)], 1, cv2.LINE_AA)
            cv2.putText(im, f'{self.classes[int(cls_)]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_palette[int(cls_)], 2, cv2.LINE_AA)

        # # Show image
        # if vis:
        #     cv2.imshow('demo', im)
        #     cv2.waitKey(1)


        # # Save image
        if save:
            print("保存图片成功")
            cv2.imwrite('demo.png', im)
        return im

def read_Video(video_path, model, conf_threshold, iou_threshold):
    cap = cv2.VideoCapture(video_path)


    while True:
        # Capture NEXT frame
        # next_frame = player.next()
        start_time = time.time()
        frame_number = 0
        ret, img = cap.read()
        if ret:
            boxes = model(img, conf_threshold, iou_threshold)
            if len(boxes) > 0:
                img = model.draw_and_visualize(img, boxes, vis=False, save=True)
            stop_time = time.time()
            total_time = stop_time - start_time
            frame_number = frame_number + 1
            async_fps = frame_number / total_time
            print("async_fps",async_fps)
            cv2.putText(img, f'{async_fps:.3f}', (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("img", img)
            key = cv2.waitKey(1)
            # escape = 27
            if key == 27:
                break


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=r'/home/taoxifa/Ai_project/yolov8/runs/detect/train/weights/best_int8_openvino_model/best.xml', help='Path to ONNX model')
    parser.add_argument('--source', type=str, default=str(r'/home/taoxifa/Ai_project/yolov8/images/data007753.jpg'), help='Path to input image')
    parser.add_argument('--imgsz', type=tuple, default=(640, 640), help='Image input size')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--infer_tool', type=str, default='openvino', choices=("openvino", "onnxruntime"),
                        help='选择推理引擎')
    args = parser.parse_args()

    # Build model
    model = YOLOv8(args.model, args.imgsz, args.infer_tool)

    # Read image by OpenCV
    img = cv2.imread(args.source)
    # 视频推理
    # video_path = r"E:\date_collection\安全帽\工地全景视频.mp4"
    # read_Video(video_path, model, conf_threshold=args.conf, iou_threshold=args.iou)  # 视频推理


    # Inference
    boxes = model(img, conf_threshold=args.conf, iou_threshold=args.iou) # 图片推理

    # Visualize
    if len(boxes) > 0:
        model.draw_and_visualize(img, boxes, vis=False, save=True)
    # cv2.destroyAllWindows()


# 图片检测、视频检测测试
import cv2
import numpy as np
from detection import YOLODet
import gradio as gr

model = 'yolov8m.onnx'
yolo_det = YOLODet(model, conf_thres=0.5, iou_thres=0.3)

def det_img(cv_src):
    yolo_det(cv_src)
    cv_dst = yolo_det.draw_detections(cv_src)

    return cv_dst

def detectio_video(input_path,model_path,output_path):

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(5))

    t = int(1000 / fps)

    videoWriter = None

    det = YOLODet(model_path, conf_thres=0.3, iou_thres=0.5)

    while True:

        # try:
        _, img = cap.read()
        if img is None:
            break

        det(img)

        cv_dst = det.draw_detections(img)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(output_path, fourcc, fps, (cv_dst.shape[1], cv_dst.shape[0]))
            videoWriter.write(cv_dst)
        
        cv2.imshow("detection", cv_dst)
        cv2.waitKey(t)

        if cv2.getWindowProperty("detection", cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

     input = gr.Image()
     output = gr.Image()

     demo = gr.Interface(fn=det_img, inputs=input, outputs=output)
     demo.launch()

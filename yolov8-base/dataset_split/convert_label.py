# 将LabelMe标注生成的JSON文件转换为YOLO格式的TXT文件需要进行一些数据格式的映射和坐标的转换
# 遍历LabelMe JSON文件中的每个对象，提取类别和边界框的坐标。
# 将坐标转换为相对于图像宽度和高度的相对值。
# 将转换后的信息写入YOLO格式的TXT文件。

import os
import numpy as np
import json
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from os import getcwd

def get_file(json_path,test):
    files = glob(json_path + "*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    if test:
        trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
    else:
        trainval_files = files
        test_files = []

    return trainval_files, test_files

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

#
# print(wd)

def json_to_txt(json_path,files, txt_name,classes):
    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')
    list_file = open('tmp/%s.txt' % (txt_name), 'w')
    for json_file_ in files:
        # print(json_file_)
        json_filename = json_path + json_file_ + ".json"
        imagePath = json_path + json_file_ + ".jpg"
        list_file.write('%s/%s\n' % (wd, imagePath))
        out_file = open('%s/%s.txt' % (json_path, json_file_), 'w')
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread(json_path + json_file_ + ".jpg").shape

        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
            xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
            ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
            ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
            label = multi["label"]

            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                cls_id = classes.index(label)
                print(json_file_)
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert((width, height), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                # print(json_filename, xmin, ymin, xmax, ymax, cls_id)


if __name__ == '__main__':
    #  是标注好的数据所在的目标，里面包含了原始数据和标签json文件，当运行完上面的脚本之后，在当前目录下会多出与每个图像名称相同的txt文件。
    wd = getcwd()
    classes_name = ["person","sack", "elec", "bag", "box", "um", "caron", "boot","pail"]
    path = "xxxx/images/"
    train_file,test_file = get_file(path,False)
    json_to_txt(path,train_file,"train",classes_name)

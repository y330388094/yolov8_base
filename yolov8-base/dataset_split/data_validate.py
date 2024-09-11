# 转换完成之后，要验证数据转换是否正确：

import  cv2
import os
def Xmin_Xmax_Ymin_Ymax(img_path, txt_path):
    """
    :param img_path: 图片文件的路径
    :param txt_path: 标签文件的路径
    :return:
    """
    img = cv2.imread(img_path)
    # 获取图片的高宽
    h,w, _ = img.shape

    con_rect = []
    # 读取TXT文件 中的中心坐标和框大小
    with open(txt_path, "r") as fp:
        # 以空格划分
        lines =fp.readlines()
        for l in lines:
            contline= l.split(' ')

            xmin = float((contline[1])) - float(contline[3]) / 2
            xmax = float(contline[1]) + float(contline[3]) / 2
            ymin = float(contline[2]) - float(contline[4]) / 2
            ymax = float(contline[2].strip()) + float(contline[4].strip()) / 2
            xmin, xmax = w * xmin, w * xmax
            ymin, ymax = h * ymin, h * ymax

            con_rect.append((contline[0], xmin, ymin, xmax, ymax))

    return con_rect
#根据label坐标画出目标框
def plot_tangle(img_dir,txt_dir):

    contents = os.listdir(img_dir)

    for file in contents:
        img_path = os.path.join(img_dir,file)
        img = cv2.imread(img_path)
        txt_path = os.path.join(txt_dir,(os.path.splitext(os.path.basename(file))[0] + ".txt"))

        con_rect =  Xmin_Xmax_Ymin_Ymax(img_path, txt_path)

        for rect in con_rect:
            cv2.rectangle(img, (int(rect[1]), int(rect[2])), (int(rect[3]), int(rect[4])), (0, 0, 255))

        cv2.namedWindow()
        cv2.imshow("src",img)
        cv2.waitKey()

if __name__=="__main__":
    img_dir = r"xxx\images"
    txt_dir = r"xxx\labels"
    plot_tangle(img_dir,txt_dir)

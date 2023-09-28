import os
import math
import shutil
import os.path as osp
import numpy as np
import xml.etree.ElementTree as et
from math import ceil
import cv2
import codecs
from PIL import Image

import shutil
from tqdm import tqdm


NAME_LABEL_MAP =  {
        '航母': 1,
        '黄蜂级': 2,
        '塔瓦拉级': 3,
        '奥斯汀级': 4,
        '惠特贝岛级': 5,
        '圣安东尼奥级': 6,
        '新港级': 7,
        '提康德罗加级': 8,
        '阿利·伯克级': 9,
        '佩里级': 10,
        '刘易斯和克拉克级': 11,
        '供应级': 12,
        '凯泽级': 13,
        '霍普级': 14,
        '仁慈级': 15,
        '自由级': 16,
        '独立级': 17,
        '复仇者级': 18,
        '潜艇':19,
        '其他':20
        }

NAME_LABEL_MAP_en =  {
       'Aircraft carriers': 1,
        'Wasp class': 2,
        'Tarawa class': 3,
        'Austin class': 4,
        'Whidbey Island class': 5,
        'San Antonio class': 6,
        'Newport class': 7,
        'Ticonderoga class ': 8,
        'Arleigh Burke class': 9,
        'Perry class': 10,
        'Lewis and Clark class': 11,
        'Supply class': 12,
        'Henry J. Kaiser class': 13,
        'Bob Hope Class': 14,
        'Mercy class': 15,
        'Freedom class': 16,
        'Independence class': 17,
        'Avenger class': 18,
        'Submarine':19,
        'Other':20
        }



class fgsd2dota():
    def __init__(self, img_path, ann_path, save_path):
        self.images_path = img_path
        self.ann_path = ann_path
        self.dataname = "rsdd"
        self.save_img_path = osp.join(save_path, "images")
        self.save_ann_path = osp.join(save_path, "annfiles")

    def xywha2xyrbox(self):

        # 计算旋转框的四个顶点坐标
        angle = mbox_ang  # 将角度转换为弧度
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # 计算旋转框的四个顶点坐标
        x1 = mbox_cx - mbox_w / 2 * cos_angle - mbox_h / 2 * sin_angle
        y1 = mbox_cy - mbox_w / 2 * sin_angle + mbox_h / 2 * cos_angle

        x2 = mbox_cx + mbox_w / 2 * cos_angle - mbox_h / 2 * sin_angle
        y2 = mbox_cy + mbox_w / 2 * sin_angle + mbox_h / 2 * cos_angle

        x3 = mbox_cx + mbox_w / 2 * cos_angle + mbox_h / 2 * sin_angle
        y3 = mbox_cy + mbox_w / 2 * sin_angle - mbox_h / 2 * cos_angle

        x4 = mbox_cx - mbox_w / 2 * cos_angle + mbox_h / 2 * sin_angle
        y4 = mbox_cy - mbox_w / 2 * sin_angle - mbox_h / 2 * cos_angle


    def run(self):
        name_count = {}
        for filename in os.listdir(self.ann_path):
            raster = os.path.join(self.ann_path,filename)
            mydoc = et.parse(raster)
            root = mydoc.getroot()

            self.items = root.findall('object')
            filename = os.path.splitext(os.path.split(filename)[-1])[0]

            for item in self.items:
                label = item.find("name").text
                robndbox = item.find("robndbox")
                cx = float(robndbox.find("cx").text)
                cy = float(robndbox.find("cy").text)
                w = float(robndbox.find("w").text)
                h = float(robndbox.find("h").text)
                angle = float(robndbox.find("angle").text)





        # 打印不同名称及其对应的计数
        for name, count in name_count.items():
            print(f"Name: {name}, Count: {count}")




if __name__ == '__main__':
    model = "fgsd"
    name = ["train","test"]
    for folder in name:
        img_path = r"D:\omq\omqdata\rgb\FGSD2021\{}\images".format(folder)
        ann_path = r"D:\omq\omqdata\rgb\FGSD2021\{}\annfiles".format(folder)
        save_path = r"D:\omq\omqdata\sar\dota\{}/{}".format(model,folder)

        r = fgsd2dota(img_path, ann_path, save_path)
        r.run()






    # folder_path = r"D:\omq\omqdata\rgb\classifier\FGSCR"
    # for folder in os.listdir(folder_path):
    #     if folder ==".DS_Store":
    #         continue
    #     image_count = 0
    #     for filename in os.listdir(os.path.join(folder_path,folder)):
    #         if any(filename.lower().endswith(ext) for ext in ".jpg"):
    #             image_count += 1
    #     print(folder[4:].replace("_"," "),image_count)
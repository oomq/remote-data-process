
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
from tqdm import tqdm



class hrsc2dota():
    def __init__(self, img_path, ann_path, imgset_path, save_path):
        self.images_path = img_path
        self.ann_path = ann_path
        self.imgset_path = imgset_path
        self.dataname = "hrsc"
        self.imageset_path= imgset_path
        self.save_img_path = osp.join(save_path, "images")
        self.save_ann_path = osp.join(save_path, "annfiles")
        os.makedirs(self.save_img_path, exist_ok=True)
        os.makedirs(self.save_ann_path, exist_ok=True)
        self.save_flag = 1
        self.rows = 2
        self.img_size = 512
        self.image_names=[]

    # def rbox2hbb(self ,boxes):
    #     # < cx > 299.8607 < / cx >
    #     # < cy > 83.7978 < / cy >
    #     # < h > 15.24122428894043 < / h >
    #     # < w > 5.936710834503174 < / w >
    #     # < angle > -0.4691457200006939 < / angle >
    #     cx, cy, w ,h, angle = [float(p.text) for p in boxes]
    #     angle_rad = angle
    #
    #     # 计算旋转后的水平框的坐标和尺寸
    #     new_w = abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad))
    #     new_h = abs(h * math.cos(angle_rad)) + abs(w * math.sin(angle_rad))
    #     new_xmin = cx - new_w / 2
    #     new_ymin = cy - new_h / 2
    #     new_xmax = cx + new_w / 2
    #     new_ymax = cy + new_h / 2
    #     return [new_xmin, new_ymin, new_xmax, new_ymax]

    def run(self):
        with open(self.imageset_path, 'r') as file:
            lines = file.readlines()

            for line in tqdm(lines):
                parts = line.strip().split()  # 拆分每一行
                if len(parts) == 2:
                    filename, action = parts
                    if action == '1':
                        source_img_path = f"{self.images_path}/{filename}.jpg"
                        target_img_path = f"{self.save_img_path}/{filename}.jpg"
                        shutil.copy(source_img_path, target_img_path)  # 复制文件
                        source_ann_path = f"{self.ann_path}/{filename}.xml"
                        target_ann_path = f"{self.save_ann_path}/{filename}.xml"
                        shutil.copy(source_ann_path, target_ann_path)  # 复制文件
                    elif action == '-1':
                        pass  # 舍弃文件（不执行任何操作）

if __name__ == '__main__':
    # model = "RSDD"
    name = ["trainval", "test"]

    for _name in name:
        print(_name)
        img_path = r"D:\omq\omqdata\rgb\TGRS-HRRSD-Dataset-mstr-gthb\OPT2017\JPEGImages"
        ann_path = r"D:\omq\omqdata\rgb\TGRS-HRRSD-Dataset-mstr-gthb\OPT2017\Annotations"
        imgset_path = r"D:\omq\omqdata\rgb\TGRS-HRRSD-Dataset-mstr-gthb\OPT2017\ImageSets\Main\ship_{}.txt".format(_name)
        save_path = r"D:\omq\omqdata\rgb\custom\hrrsd"

        r = hrsc2dota(img_path, ann_path,imgset_path,save_path)
        r.run()
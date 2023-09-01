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
class rsdd2dota():
    def __init__(self, img_path, ann_path, imgset_path, save_path):
        self.images_path = img_path
        self.ann_path = ann_path
        self.imgset_path = imgset_path
        self.dataname = "rsdd"

        self.save_img_path = osp.join(save_path, "images")
        self.save_ann_path = osp.join(save_path, "annfiles")
        os.makedirs(self.save_img_path, exist_ok=True)
        os.makedirs(self.save_ann_path, exist_ok=True)
        self.cols = 2
        self.rows = 2
        self.img_size = 512
        self.image_names=[]


    def image_compose(self,i=None):
        to_image = Image.new('RGB', (self.cols * self.img_size,
                                     self.rows * self.img_size))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(1, self.cols + 1):
            for x in range(1, self.rows + 1):
                from_image = Image.open(osp.join(self.images_path,
                                                 self.image_names[i + 2 * (y - 1) + (x - 1)])).resize(
                                            (self.img_size, self.img_size), Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * self.img_size, (y - 1) * self.img_size))
        return to_image

    def rbox2hbb(self,boxes):
        # < cx > 299.8607 < / cx >
        # < cy > 83.7978 < / cy >
        # < h > 15.24122428894043 < / h >
        # < w > 5.936710834503174 < / w >
        # < angle > -0.4691457200006939 < / angle >
        cx, cy, w,h, angle = [float(p.text) for p in boxes]
        angle_rad = angle

        # 计算旋转后的水平框的坐标和尺寸
        new_w = abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad))
        new_h = abs(h * math.cos(angle_rad)) + abs(w * math.sin(angle_rad))
        new_xmin = cx - new_w / 2
        new_ymin = cy - new_h / 2
        new_xmax = cx + new_w / 2
        new_ymax = cy + new_h / 2
        return [new_xmin, new_ymin, new_xmax, new_ymax]


    def ann_compose(self,i=None,f=None):
        for y in range(1, self.cols + 1):
            for x in range(1, self.rows + 1):
                xml_file = open(osp.join(self.ann_path,self.image_names[i+2*(y-1)+(x-1)].
                                         replace("jpg","xml")),encoding='utf-8')
                tree = et.parse(xml_file)
                root = tree.getroot()
                for obj in root.iter('object'):
                    box1 = obj.find("robndbox")
                    l,t,r,b = self.rbox2hbb(box1)
                    l = l + self.img_size*(x-1)
                    t = t + self.img_size*(y-1)
                    r = r + self.img_size*(x-1)
                    b = b + self.img_size*(y-1)
                    f.write("{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}"
                            " ship 0\n".format(l,t,l,b,r,b,r,t))

    def run(self):
        # self.image_names = [name for name in os.listdir(self.images_path)]
            with open(self.imgset_path,"r") as f:
                self.image_names = [p.replace("\n","")+".jpg" for p in f.readlines()]
            for i in range(0, len(self.image_names), 4):
                img = self.image_compose(i=i)  # 调用函数
                name = osp.join(self.save_img_path, "{:0=4d}.jpg".format(i))
                img.save(name)
                with open(osp.join(self.save_ann_path, "{:0=4d}.txt".format(i)),"w+") as f:
                    self.ann_compose(i,f)



if __name__ == '__main__':
    model = "RSDD"
    name = ["train", "test"]

    for _name in name:
        img_path = "D:\omq\omqdata\sar\RSDD\JPEGImages"
        ann_path = "D:\omq\omqdata\sar\RSDD\Annotations"
        imagesets = "D:\omq\omqdata\sar\RSDD\ImageSets\{}.txt".format(_name)
        save_path = r"D:\omq\omqdata\sar\dota\{}/{}".format(model,_name)

        r = rsdd2dota(img_path, ann_path, imagesets,save_path)
        r.run()
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
    def __init__(self, img_path, ann_path, save_path):
        self.images_path = img_path
        self.ann_path = ann_path
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
                temp_image = Image.new('RGB', (self.img_size, self.img_size))  # 创建一个新图
                from_image = Image.open(osp.join(self.images_path,
                                                 self.image_names[i + self.cols * (y - 1) + (x - 1)]))
                temp_image.paste(from_image,(0,0))

                to_image.paste(temp_image, ((x - 1) * self.img_size, (y - 1) * self.img_size))
        return to_image

    def ann_compose(self,i=None,f=None):
        for y in range(1, self.cols + 1):
            for x in range(1, self.rows + 1):
                xml_file = open(osp.join(self.ann_path,self.image_names[i+self.cols*(y-1)+(x-1)].
                                         replace("jpg","xml")),encoding='utf-8')
                tree = et.parse(xml_file)
                root = tree.getroot()
                for obj in root.iter('object'):
                    box1 = obj.find("bndbox")
                    l,t,r,b =[p.text for p in box1]
                    l = int(l) + self.img_size*(x-1)
                    t = int(t) + self.img_size*(y-1)
                    r = int(r) + self.img_size*(x-1)
                    b = int(b) + self.img_size*(y-1)
                    f.write("{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}"
                            " ship 0\n".format(l,t,l,b,r,b,r,t))

    def run(self):
        self.image_names = [name for name in os.listdir(self.images_path)]
        for i in range(0, len(self.image_names), 4):
            img = self.image_compose(i=i)  # 调用函数
            name = osp.join(self.save_img_path, "{:0=4d}.jpg".format(i))
            img.save(name)
            with open(osp.join(self.save_ann_path, "{:0=4d}.txt".format(i)),"w+") as f:
                self.ann_compose(i,f)



if __name__ == '__main__':
    model = "SSDD"
    folder="train"
    img_path = r"D:\omqdata\sar\Official-SSDD-OPEN\BBox_SSDD\voc_style\JPEGImages_{}".format(folder)
    ann_path = r"D:\omqdata\sar\Official-SSDD-OPEN\BBox_SSDD\voc_style\Annotations_{}".format(folder)
    save_path = r"D:\omqdata\sar\dota\{}/".format(model)

    r = rsdd2dota(img_path, ann_path, save_path)
    r.run()
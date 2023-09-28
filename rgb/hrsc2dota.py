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
        # self.image_names = [name for name in os.listdir(self.images_path)]
        for file in tqdm(os.listdir(self.ann_path)):
            with open(osp.join(self.save_ann_path, \
                               file.replace("xml","txt")), "w+") as f:
                xml_file = open(osp.join(self.ann_path, file), encoding='utf-8')
                tree = et.parse(xml_file)
                root = tree.getroot()
                for obj in root.iter('HRSC_Objects'):
                    box = obj.find("HRSC_Object")
                    if box is None:
                        self.save_flag = 0
                        continue
                    box1=[
                        box.find("mbox_cx"),
                        box.find("mbox_cy"),
                        box.find("mbox_w"),
                        box.find("mbox_h"),
                        box.find("mbox_ang")
                    ]
                    mbox_cx = float(box.find("mbox_cx").text)
                    mbox_cy = float(box.find("mbox_cy").text)
                    mbox_w = float(box.find("mbox_w").text)
                    mbox_h = float(box.find("mbox_h").text)
                    mbox_ang = float(box.find("mbox_ang").text)

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

                    f.write("{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}"
                        " ship 0\n".format(x1,y1,x2,y2,x3,y3,x4,y4))

            if self.save_flag==1:
                shutil.copy(osp.join(self.images_path,file.replace("xml","bmp") ),
                        osp.join(self.save_img_path, file.replace("xml","jpg")))
            self.save_flag = 1

if __name__ == '__main__':
    # model = "RSDD"
    # name = ["train", "test"]
    #
    # for _name in name:
    img_path = r"D:\omq\omqdata\rgb\HRSC2016\HRSC2016\Test\AllImages"
    ann_path = r"D:\omq\omqdata\rgb\HRSC2016\HRSC2016\Test\Annotations"
    save_path = r"D:\omq\omqdata\rgb\custom\HRSC\test"
    # save_path = r"D:\omq\omqdata\sar\dota\{}/{}".format(model,_name)

    r = hrsc2dota(img_path, ann_path,None,save_path)
    r.run()
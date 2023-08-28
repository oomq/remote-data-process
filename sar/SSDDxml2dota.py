import os
import math
import shutil
import os
import os.path as osp
import numpy as np
import xml.etree.ElementTree as et

class xml2dota():
    def __init__(self,img_path,ann_path,save_path):
        self.images_path= img_path
        self.ann_path = ann_path
        self.dataname = "ssdd"
        self.save_img_path = osp.join(save_path,"images")
        self.save_ann_path = osp.join(save_path, "annfiles")
        os.makedirs(self.save_img_path, exist_ok=True)
        os.makedirs(self.save_ann_path, exist_ok=True)
    def run(self):
        for xml in os.listdir(self.ann_path):
            xml_file =open(osp.join(self.ann_path,xml),encoding='utf-8')
            tree = et.parse(xml_file)
            root = tree.getroot()
            with open("{}/{}.txt".format(self.save_ann_path,
                                         osp.splitext(xml)[0]
                                        ),"w+") as f:
                for obj in root.iter('object'):
                    box1 = obj.find("rotated_bndbox")
                    for xy in box1[5:]:
                        f.write("{} ".format(xy.text))
                    f.write('ship 0\n')
            shutil.copy(osp.join(self.images_path,osp.splitext(xml)[0]+".jpg"),
                        osp.join(self.save_img_path,"{}{}.jpg".format(self.dataname,osp.splitext(xml)[0])))

if __name__ == '__main__':
    model = "test"
    img_path = r"D:\omqdata\sar\Official-SSDD-OPEN\RBox_SSDD\voc_style\JPEGImages_{}".format(model)
    ann_path = r"D:\omqdata\sar\Official-SSDD-OPEN\RBox_SSDD\voc_style\Annotations_{}".format(model)
    save_path = r"D:\omqdata\sar\dota\{}".format(model)

    r = xml2dota(img_path,ann_path,save_path)
    r.run()
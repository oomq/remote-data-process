import os
import math
import shutil
import os.path as osp
import numpy as np
import xml.etree.ElementTree as et
from math import ceil
import cv2
import codecs
import datetime
import itertools
import json
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import time
import os
try:
    import shapely.geometry as shgeo
except ImportError:
    shgeo = None

try:###
    from osgeo import gdal
except ImportError:
    gdal = None
from PIL import Image

class sarshiphugepic2dota():
    def __init__(self,img_path,ann_path,save_path):
        self.img_path= img_path
        self.ann_path = ann_path
        self.dataname = "ssdd"
        self.save_img_path = osp.join(save_path,"images")
        self.save_ann_path = osp.join(save_path, "annfiles")
        os.makedirs(self.save_img_path, exist_ok=True)
        os.makedirs(self.save_ann_path, exist_ok=True)
        self.width = 0
        self.height = 0
        self.sizes = [1000]
        self.gaps = [0]
        self.img_rate_thr = 0.6
        self.iof_thr=0.7
        self.padding_value = [104,116,124]
        self.no_padding= False
        self.img_ext = ".jpg"



    def get_sliding_window(self):
        eps = 0.01
        windows = []
        width, height = self.width,self.height
        sizes,gaps = self.sizes,self.gaps
        for size, gap in zip(sizes, gaps):
            assert size > gap, f'invaild size gap pair [{size} {gap}]'
            step = size - gap  ###ss:1024-200

            x_num = 1 if width <= size else ceil((width - size) / step + 1)
            x_start = [step * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size > width:
                if width - x_start[-1] < 500:  ###
                    x_start = x_start[:-1]
                else:
                    x_start[-1] = width - size

            y_num = 1 if height <= size else ceil((height - size) / step + 1)
            y_start = [step * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size > height:
                if height - y_start[-1] < 500:  ###
                    y_start = y_start[:-1]
                else:
                    y_start[-1] = height - size

            start = np.array(
                list(itertools.product(x_start, y_start)), dtype=np.int64)
            stop = start + size
            windows.append(np.concatenate([start, stop], axis=1))
        windows = np.concatenate(windows, axis=0)

        img_in_wins = windows.copy()
        img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
        img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
        img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                    (img_in_wins[:, 3] - img_in_wins[:, 1])
        win_areas = (windows[:, 2] - windows[:, 0]) * \
                    (windows[:, 3] - windows[:, 1])
        img_rates = img_areas / win_areas
        if not (img_rates > self.img_rate_thr).any():
            max_rate = img_rates.max()
            img_rates[abs(img_rates - max_rate) < eps] = 1
        return windows[img_rates > self.img_rate_thr]

    def hbb2poly(self,boxes):
        polys=[]
        for box in boxes:
            l,t,r,b = box
            polys.append([l,t,r,t,r,b,l,b])
        return np.array(polys).astype(int)

    def translate(self,bboxes, x, y):

        dim = bboxes.shape[-1]
        translated = bboxes + np.array([x, y] * int(dim / 2), dtype=np.float32)
        return translated


    def bbox_overlaps_iof(self, bboxes1, bboxes2, eps=1e-6):
        bboxes1 = np.array(bboxes1).astype(int)
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]

        if rows * cols == 0:
            return np.zeros((rows, cols), dtype=np.float32)

        hbboxes1 = bboxes1
        hbboxes2 = bboxes2
        hbboxes1 = hbboxes1[:, None, :]
        lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
        rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
        wh = np.clip(rb - lt, 0, np.inf)
        h_overlaps = wh[..., 0] * wh[..., 1]

        l, t, r, b = [bboxes2[..., i] for i in range(4)]
        polys2 = np.stack([l, t, r, t, r, b, l, b], axis=-1)
        if shgeo is None:
            raise ImportError('Please run "pip install shapely" '
                              'to install shapely first.')
        bboxes1 = self.hbb2poly(bboxes1)
        sg_polys1 = [shgeo.Polygon(p) for p in bboxes1.reshape(rows, -1, 2)]
        sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]
        overlaps = np.zeros(h_overlaps.shape)
        for p in zip(*np.nonzero(h_overlaps)):
            overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
        unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
        unions = unions[..., None]

        unions = np.clip(unions, eps, np.inf)
        outputs = overlaps / unions
        if outputs.ndim == 1:
            outputs = outputs[..., None]
        return outputs

    def get_window_obj(self,boxes, windows, iof_thr):
        iofs = self.bbox_overlaps_iof(boxes, windows)
        window_anns = []
        for i in range(windows.shape[0]):
            win_iofs = iofs[:, i]
            pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()

            win_ann = dict()
            # for k, v in info['ann'].items():
            #     try:
            #         win_ann[k] = v[pos_inds]
            #     except TypeError:
            #         win_ann[k] = [v[i] for i in pos_inds]
            win_ann["bboxes"] = np.array(self.hbb2poly(boxes)[pos_inds],dtype=int)
            win_ann["bboxes"]

            win_ann['trunc'] = win_iofs[pos_inds] < 1
            window_anns.append(win_ann)
        return window_anns

    def split_sigle_img(self,windows, window_anns):
        patch_infos = []
        for i in range(windows.shape[0]):
            patch_info = dict()

            window = windows[i]
            x_start, y_start, x_stop, y_stop = window.tolist()
            patch_info['x_start'] = x_start
            patch_info['y_start'] = y_start
            patch_info['id'] = \
                self.filename+'__' + str(x_stop - x_start) + \
                '__' + str(x_start) + '___' + str(y_start)

            ann = window_anns[i]
            ann['bboxes'] = self.translate(ann['bboxes'], -x_start, -y_start)
            patch_info['ann'] = ann
            # print(ann)

            patch = self.img_file[y_start:y_stop, x_start:x_stop]
            if not self.no_padding:
                height = y_stop - y_start
                width = x_stop - x_start
                if height > patch.shape[0] or width > patch.shape[1]:
                    padding_patch = np.empty((height, width, patch.shape[-1]),
                                             dtype=np.uint8)
                    if not isinstance(self.padding_value, (int, float)):
                        assert len(self.padding_value) == patch.shape[-1]
                    padding_patch[...] = self.padding_value
                    padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                    patch = padding_patch
            patch_info['height'] = patch.shape[0]
            patch_info['width'] = patch.shape[1]

            bboxes_num = patch_info['ann']['bboxes'].shape[0]
            outdir = os.path.join(self.save_ann_path, patch_info['id'] + '.txt')
            patch_info['filename'] = patch_info['id'] + self.img_ext
            patch_infos.append(patch_info)

            ### pass non-ship picture
            if bboxes_num == 0 :
                continue

            with codecs.open(outdir, 'w', 'utf-8') as f_out:
                if bboxes_num == 0:
                    pass
                else:
                    for idx in range(bboxes_num):
                        obj = patch_info['ann']
                        outline = ' '.join(list(map(str, obj['bboxes'][idx])))
                        outline = outline + ' ship 0'
                        f_out.write(outline + '\n')

            cv2.imwrite(osp.join(self.save_img_path, patch_info['filename'] ), patch)
        return patch_infos

    def percentage_truncation(self,im_data, lower_percent=0.001, higher_percent=99.999, per_channel=True):
        '''
        :param im_data: 图像矩阵(h, w, c)
        :type im_data: numpy
        :param lower_percent: np.percentile的最低百分位
        :type lower_percent: float
        :param higher_percent: np.percentile的最高百分位
        :type higher_percent: float
        :return: 返回图像矩阵(h, w, c)
        :rtype: numpy
        '''

        a = 0  # 最小值
        b = 255  # 最大值
        c = np.percentile(im_data, lower_percent)
        d = np.percentile(im_data, higher_percent)
        out = a + (im_data - c) * (b - a) / (d - c)
        out = np.clip(out, a, b).astype(np.uint8)
        return out



    def run(self):
        for xml in os.listdir(self.ann_path):
            print(xml)
            self.filename= osp.splitext(xml)[0]
            xml_file =open(osp.join(self.ann_path,xml),encoding='utf-8')
            # self.img_file = cv2.imread(osp.join(self.img_path, self.filename.replace("-label",".tiff")))
            # self.img_file = cv2.cvtColor(self.img_file, cv2.COLOR_BGR2GRAY)
            # self.img_file = cv2.equalizeHist(self.img_file)

            self.img_file = gdal.Open(osp.join(self.img_path,
                                               self.filename.replace("-label", ".tiff")),gdal.GA_ReadOnly)

            if self.img_file is None:
                raise Exception(f"Unable to open file: tiff")

            image_data = self.img_file.ReadAsArray()
            image_data = np.array(image_data, dtype=np.uint16)
            ## >255 ?= 255
            image_data = np.clip(image_data, None, 255)
            image_data = self.percentage_truncation(image_data)
            self.img_file = cv2.equalizeHist(image_data)
            # 读取影像数据
            # image_data = self.img_file.ReadAsArray()
            # image_data = np.array(image_data, dtype=np.uint16)
            #
            # # 对影像数据进行直方图均衡化
            # self.img_file = cv2.equalizeHist(image_data)

            tree = et.parse(xml_file)
            root = tree.getroot()
            # self.width = int(root.find("size").find("width").text)
            # self.height = int(root.find("size").find("height").text)
            self.width =3000 ###xml格式有问题
            self.height = 3000
            boxes= []
            for obj in root.iter('object'):
                box = [x.text for x in obj.find("bndbox")]
                boxes.append(box)
            windows = self.get_sliding_window()
            window_anns = self.get_window_obj(boxes, windows, self.iof_thr)
            self.split_sigle_img(windows,window_anns)




if __name__ == '__main__':
    model = "SARSHIP"
    img_path = r"D:\omqdata\sar\SARship-1\images"
    ann_path = r"D:\omqdata\sar\SARship-1\annfiles"
    save_path = r"D:\omqdata\sar\dota\{}/".format(model)

    r = sarshiphugepic2dota(img_path,ann_path,save_path)
    r.run()
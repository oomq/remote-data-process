import os
import os.path as osp
import cv2
import codecs
import random
import numpy as np
import itertools
from math import ceil
from PIL import Image
import matplotlib.pyplot as plt
import pathlib

# import geopandas as gpd
from tqdm import tqdm

from osgeo import gdal, ogr
import shapefile

class crop():
    def __init__(self,raster,r,imgname):
        self.images_path = "txt2shp/images"
        self.shp_path = "txt2shp/shp"
        self.save_dir = r'shp2falseexample/'
        self.save_dir_images = r'shp2falseexample/images/'
        self.save_dir_txt = r'shp2falseexample/txt/'
        os.makedirs(self.save_dir_images, exist_ok=True)
        os.makedirs(self.save_dir_txt, exist_ok=True)
        self.maxtarget_height=0
        self.max_width = 1024
        self.max_height=1024
        self.big_image = Image.new('RGB', (self.max_width, self.max_height))
        self.filename = imgname
        self.current_width = 0
        self.current_height = 0
        self.maxtarget_height = 0
        self.raster = raster
        self.shp = r
        self.img_ext = ".png"
        self.windows =[]
        self.overcounter =0
        self.saveflag = 0

    def get_sliding_window(self,shape, sizes, gaps, img_rate_thr):
        """Get sliding windows.

        Args:
            info (dict): Dict of image's width and height.
            sizes (list): List of window's sizes.
            gaps (list): List of window's gaps.
            img_rate_thr (float): Threshold of window area divided by image area.

        Returns:
            list[np.array]: Information of valid windows.
        """
        eps = 0.01
        windows = []
        width, height = shape[0],shape[1]
        for size, gap in zip(sizes, gaps):
            assert size > gap, f'invaild size gap pair [{size} {gap}]'
            step = size - gap

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
                if height - y_start[-1] <500:###
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
        if not (img_rates > img_rate_thr).any():
            max_rate = img_rates.max()
            img_rates[abs(img_rates - max_rate) < eps] = 1
        return windows[img_rates > img_rate_thr]

    def trans_shpgeo2xy(self,lay):
        geom = lay.GetGeometryRef()  # 提取该要素的轮廓坐标
        # 对多边形Geometry格式进行字符串裁剪，截取并放入geom_str的list中
        geom_replace = str(geom).replace('(', '')  # 首先需将Geometry格式转换为字符串
        geom_replace = geom_replace.replace(')', '')
        geom_replace = geom_replace.replace(' ', ',')
        geom_str = geom_replace.split(',')[1:]  # Geometry格式中首个字符串为POLYGON，需跳过，故从1开始

        # print(geom_str)  # 打印geom_str List
        geom_x = geom_str[0::2]  # 在list中输出经度坐标
        # print(geom_str[0::2])
        geom_y = geom_str[1::2]  # 在list中输出纬度坐标
        # print(geom_str[1::2])
        TransformPara = self.TransformPara
        polygon = []  # (n, 2)
        '''对每个坐标进行转换'''
        for j in range(len(geom_x)):
            # print('----', geom_x[j], geom_y[j])
            dTemp = TransformPara[1] * TransformPara[5] - TransformPara[2] * TransformPara[4]
            Xpixel = (TransformPara[5] * (float(geom_x[j]) - TransformPara[0]) - TransformPara[2] * (
                    float(geom_y[j]) - TransformPara[3])) / dTemp + 0.5
            Yline = (TransformPara[1] * (float(geom_y[j]) - TransformPara[3]) - TransformPara[4] * (
                    float(geom_x[j]) - TransformPara[0])) / dTemp + 0.5

            col = [round(Xpixel, 2), round(Yline, 2)]  # 接收行列坐标数据
            polygon.append(col)  # 放入row中


        return polygon


    def combine_images(self,img):
        img_data = Image.fromarray(img.astype(np.uint8))
        target_width, target_height = img_data.size

        ##Determine whether it is out of big_images range
        if self.current_width + target_width < self.max_width and self.current_height+target_height<self.max_height:#没超出x轴但没有超y轴
            # 在大图中换行放图
            self.big_image.paste(img_data, (self.current_width, self.current_height))
            self.current_width += target_width
            self.maxtarget_height = target_height+self.current_height if target_height+self.current_height > self.maxtarget_height else self.maxtarget_height
            # print("1",self.current_width,self.current_height,self.maxtarget_height)
            # plt.imshow(self.big_image)
            # plt.show()


        elif self.current_width + target_width > self.max_width and self.current_height+target_height<self.max_height:#超x但没有超y轴
            self.current_width = 0
            self.big_image.paste(img_data, (0, self.maxtarget_height))
            self.current_width += target_width
            self.current_height = self.maxtarget_height  ##update y again
            self.maxtarget_height = target_height+self.current_height if target_height+self.current_height > self.maxtarget_height else self.maxtarget_height

            # print("2", self.current_width, self.current_height, self.maxtarget_height)
            # plt.imshow(self.big_image)
            # plt.show()


        else:#超y 或者都超
            self.big_image.save(osp.join(self.save_dir_images,self.filename+ "{}.png".format(self.overcounter)) )
            self.create_none_txt(osp.join(self.save_dir_txt,self.filename+ "{}.txt".format(self.overcounter)))
            self.overcounter +=1
            self.big_image = Image.new('RGB', (self.max_width, self.max_height))
            self.big_image.paste(img_data, (0, 0))
            self.current_width = target_width
            self.current_height = 0
            self.maxtarget_height = target_height
            self.saveflag = 1

    def crop_and_save_img(self,img,no_padding,padding_value):

        patch_infos = []

        for i in range(self.windows.shape[0]):
            window = self.windows[i]
            x_start, y_start, x_stop, y_stop = window.tolist()


            patch = img[y_start:y_stop, x_start:x_stop]
            if not no_padding:
                height = y_stop - y_start
                width = x_stop - x_start
                if height > patch.shape[0] or width > patch.shape[1]:
                    padding_patch = np.empty((height, width, patch.shape[-1]),
                                             dtype=np.uint8)
                    if not isinstance(padding_value, (int, float)):
                        assert len(padding_value) == patch.shape[-1]
                    padding_patch[...] = padding_value
                    padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                    patch = padding_patch
            filename1 = self.filename+"_{}_{}".format(x_start,y_start)
            # print(filename1)
            cv2.imwrite(osp.join(self.save_dir_images, filename1 + self.img_ext), patch)
            self.create_none_txt(osp.join(self.save_dir_txt, filename1 + ".txt"))

        return patch_infos

    def create_none_txt(self,file_name):
        pathlib.Path(file_name).touch()

    def run(self):
            # 同时载入gdal库的图片从而获取geotransform
            srcImage = gdal.Open(self.raster)
            self.TransformPara = srcImage.GetGeoTransform()
            # 使用PyShp库打开shp文件

            #####################################3##
            img = cv2.imread(self.raster)
            if img is None:
                img = np.einsum("ijk->jki",srcImage.ReadAsArray())


            ################################################
            driver = ogr.GetDriverByName('ESRI Shapefile')  # 载入驱动
            dataSource = driver.Open(r, 0)  # 第二个参数为0是只读，为1是可写
            layer = dataSource.GetLayer(0)  # 读取第一个图层
            self.overcounter=0
            for idx, lay in enumerate(layer):
                filed = lay.GetField("class_id")
                # print(filed)
                shape = []
                if filed == 1: ###负样本裁剪
                    polygon = self.trans_shpgeo2xy(lay)
                    # print(polygon)
                    mat = np.array(polygon)
                    x1,y1 = mat.min(axis=0)
                    x2,y2 = mat.max(axis=0)
                    # print(x1,y1,x2,y2)
                    shape.append(x2-x1)
                    shape.append(y2-y1)
                    crop_img = img[int(y1):int(y2), int(x1):int(x2)]

                    self.windows = self.get_sliding_window(shape, [1024], [200], 0.3)
                    if crop_img.shape[0] < 1000 or crop_img.shape[1] < 1000:
                        self.combine_images(crop_img,)


                    else:
                        self.crop_and_save_img(crop_img,
                                               no_padding= True,padding_value=[104,116,124])

            if self.saveflag ==1 :
                self.big_image.save(osp.join(self.save_dir_images,self.filename+ "222.jpg"))
                self.create_none_txt(osp.join(self.save_dir_txt,self.filename+ "222.txt"))

if __name__ == '__main__':
    images_path = "txt2shp/images"
    shp_path = "txt2shp/shp"
    output_img = r'shp2falseexample/images/'
    output_txt = r"shp2falseexample/txt/"
    for imgname in os.listdir(images_path):
        # if not imgname == "Singapore_r0.tif":
        #     continue
        if not imgname.endswith("tif"):
            continue

        raster = osp.join(images_path, imgname)
        r = osp.join(shp_path, osp.splitext(imgname)[0] + ".shp")
        if not osp.exists(r):
            continue
        print(r)
        re = crop(raster,r,osp.splitext(imgname)[0])
        re.run()

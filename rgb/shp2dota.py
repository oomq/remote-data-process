import os
import os.path as osp
import cv2
import random
import numpy as np
# import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal, ogr
import shapely.geometry as shgeo
from skimage import segmentation, measure, morphology, color

def trans_shp_geo_to_xy(Geoshp, Geotif, field):
    '''
        地理坐标转像素坐标
    :param Geoshp: shp文件路径
    :param Geotif: tif文件路径
    :return: 转换好的像素坐标系坐标
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')  # 载入驱动
    dataSource = driver.Open(Geoshp, 0)  # 第二个参数为0是只读，为1是可写
    layer = dataSource.GetLayer(0)  # 读取第一个图层

    '''读出上下左右边界，坐标系为地理坐标系'''
    extent = layer.GetExtent()
    print('extent:', extent)
    print('ul:', extent[0], extent[1])  # 左右边界
    print('lr:', extent[2], extent[3])  # 下上边界

    n = layer.GetFeatureCount()  # 该图层中有多少个要素
    print('feature count:', n)

    '''左上角地理坐标'''
    TransformPara = gdal.Open(Geotif).GetGeoTransform() # 获取变换参数

    '''循环遍历所有的该图层中所有的要素'''
    objects = []  # 存储转换出的行列坐标的数组
    for i in range(n):
        feat = layer.GetNextFeature()  # 读取下一个要素
        class_id = feat.GetField(field)
        # if class_id == 0:
        #     continue
        # else:
        #     class_id -= 1
        # # -------------------------------
        # assert class_id in [0, 1, 2, 3, 4]
        # -------------------------------
        geom = feat.GetGeometryRef()  # 提取该要素的轮廓坐标
        # print(i, ":")
        # print(geom)     # 输出的多边形轮廓坐标

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
            # print("col:",col)
            polygon.append(col)  # 放入row中
            # print("row:", row)
        objects.append((class_id, polygon))  # 将每一行的坐标放入存储像素坐标系的数组中

    return objects

if __name__ == "__main__":
    inshp_dir="txt2shp/shp"
    inimg_dir="txt2shp/images"
    outdota_dir = "shp2dota"
    os.makedirs(outdota_dir,exist_ok=True)
    FIELD = 'class_id'


    for fid in os.listdir(inshp_dir):
        if not osp.splitext(fid)[1] ==".shp":
            continue
        fid = osp.splitext(fid)[0]
        shp_path = os.path.join(inshp_dir, fid + '.shp')
        tif_path = os.path.join(inimg_dir, fid + '.tif')
        objects = trans_shp_geo_to_xy(shp_path, tif_path, field=FIELD)
        post_objs = []
        for cls_id, poly in objects:
            if cls_id==1:
                continue
            print(cls_id)
            post_obj=[]
            inst_poly = shgeo.Polygon(poly)
            min_rect = inst_poly.minimum_rotated_rectangle
            for x,y in zip(min_rect.boundary.xy[0][:4],min_rect.boundary.xy[1][:4]):
                post_obj.append([int(x), int(y)])
            post_objs.append(post_obj)
            # print(min_rect)

            # cnt = np.array(poly)  # 必须是array数组的形式
            # rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            # box = cv2.BoxPoints(rect)
        txt_path = osp.join(outdota_dir,fid+".txt")
        with open(txt_path,"w") as f:
            for roratexy in post_objs:
                f.write(str(roratexy).replace("[","").replace("]","").replace(",","")
                            + " ship 1\n")




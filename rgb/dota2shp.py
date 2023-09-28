import os
import os.path as osp
import sys
import shapely

from tqdm import tqdm
import numpy as np
import cv2 as cv
from skimage import transform
from osgeo import osr, ogr, gdal

'''
读取tif图片获取geo格式并将DOTA格式的文件转为shp文件
'''
def trans_xy_to_geo(poly_points, trans_para):
    geo_points = []
    for num in range(2,len(poly_points)+1,2):
        x, y =poly_points[num-2:num]
        '''转换公式'''
        geo_x = trans_para[0] + float(x) * trans_para[1] + float(y) * trans_para[2]
        geo_y = trans_para[3] + float(y) * trans_para[5] + float(x) * trans_para[4]
        geo_points.append([geo_x, geo_y])
    return geo_points


# 写入shp文件,polygon
def write_polygons_to_shp(objects, tif_file, save_path):
    # 支持中文路径
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 属性表字段支持中文
    # gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(save_path)
    if ds == None:
        return "创建文件失败" + save_path
    img = gdal.Open(tif_file, gdal.GA_ReadOnly)
    if img == None:
        raise Exception("Unable to read the tif file")
    proj = img.GetProjectionRef()
    geotrans = img.GetGeoTransform()

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    srs.ImportFromWkt(proj)
    # 线：ogr.wkbLineString
    # 点：ogr.wkbPoint
    # 面：ogr.wkbPolygon
    layer = ds.CreateLayer("Polygon", srs, ogr.wkbPolygon)
    field = ogr.FieldDefn("class_id", ogr.OFTInteger)
    layer.CreateField(field)
    field = ogr.FieldDefn("score", ogr.OFTReal)
    layer.CreateField(field)
    defn = layer.GetLayerDefn()

    # tbar = tqdm(objects)
    # tbar.set_description('Write shp')

    for object in objects:
        object = object.split()
        polygon= [int(p) for p in object[:8]]
        cls = object[8]
        score = float(object[9])
        geo_poly = trans_xy_to_geo(polygon, trans_para=geotrans)
        geo_poly.append(geo_poly[0])##close the contour
        js_obj = dict(
            type='Polygon',
            coordinates=[geo_poly]
        )
        feature = ogr.Feature(defn)
        feature.SetField("class_id", cls)
        feature.SetField("score", score)
        geo_polygon = ogr.CreateGeometryFromJson(str(js_obj))
        feature.SetGeometry(geo_polygon)
        layer.CreateFeature(feature)
    del ds
    del img
    return "done"


if __name__ == "__main__":
    # fdir = r"txt2shp"
    # save_path = "txt2shp"
    # os.makedirs(save_path,exist_ok=True)
    # txt_path = osp.join(fdir,"txt1")
    # for file in os.listdir(txt_path):
    #     save_fpath = osp.join(save_path,osp.splitext(file)[0] + ".shp")
    #     ftxtpath = osp.join(txt_path, file)
    #     fpath = osp.join(fdir,"images", osp.splitext(file)[0] + ".tif")
    #     print(fpath,ftxtpath)
    #     with open(ftxtpath,"r") as f:
    #         all_dets=f.readlines()
    #         write_polygons_to_shp(all_dets, tif_file=fpath, save_path=save_fpath)


    ###single file
    fdir = "txt2shp"
    save_path = "txt2shp"
    file = "GF2_20200116_HaiNan_r2.txt"
    ftxtpath = osp.join(fdir,file)

    fpath = osp.join(fdir, "images", osp.splitext(file)[0] + ".tif")
    print(fpath)
    save_fpath =osp.join(save_path,osp.splitext(file)[0] + ".shp")
    with open(ftxtpath, "r") as f:
        all_dets = f.readlines()
        write_polygons_to_shp(all_dets, tif_file=fpath, save_path=save_fpath)

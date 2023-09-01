import dota_utils as util
import os
import cv2
import json


# wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

wordname_16 =["ship",]
def DOTA2COCO(srcpath, destfile,image_id):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'annfiles')

    data_dict = {}
    info = {'contributor': 'captain group',
           'data_created': '2019',
           'description': 'This is 1.5 version of DOTA dataset.',
           'url': 'http://captain.whu.edu.cn/DOTAweb/',
           'version': '1.5',
           'year': 2019}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_16):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    # image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.jpg')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                # single_obj['category_id'] = wordname_16.index(obj['name']) + 1
                single_obj['category_id'] = 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                # single_obj['bbox'] = obj['poly']
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)
    return image_id

if __name__ == '__main__':
    root= r'D:\omq\omqdata\sar\dota'
    name = ["train","test"]
    for _name in name:
        image_id = 1
        for folder in os.listdir(root):
            if os.path.exists(os.path.join(root,folder,_name)):
                print(folder,_name)
                image_id = DOTA2COCO(os.path.join(root,folder,_name),
                          os.path.join(root,folder,"{}.json".format(_name)),image_id=image_id+1)
                print(image_id)



    # DOTA2COCO(r'D:\omq\omqdata\sar\dota\DSSDD\{}'.format(folder),
    #           r'D:\omq\omqdata\sar\dota\DSSDD/{}.json'.format(folder))
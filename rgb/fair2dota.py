from xml.etree import ElementTree as ET
import os
import shutil
from tqdm import tqdm


'''
将FAIR1M数据集转化成DOTA格式
'''


class fair2dota():
    def __init__(self):
        self.items = 'Boeing 737, Boeing 777, Boeing 747, Boeing 787, Airbus A320, Airbus A321, Airbus A220, Airbus A330, \
                Airbus A350, COMAC C919, COMAC ARJ21, other-airplane, passenger ship, motorboat, fishing boat, \
                tugboat, engineering ship, liquid cargo ship, dry cargo ship, warship, other-ship, small car, bus, cargo truck, \
                dump truck, van, trailer, tractor, truck tractor, excavator, other-vehicle, baseball field, basketball court, \
                football field, tennis court, roundabout, intersection, bridge'
        self.items = [item.strip() for item in self.items.split(',')]
        self.convert_options = {}
        for item in self.items:
            if "boat" in item or "ship" in item :
                self.convert_options[item] ="ship"
                # self.convert_options["motorboat"] = "ignore"
            else:
                # print("Skipping ", item)
                self.convert_options[item.replace('Airbus ', '').lower()] = 'ignore'
                self.convert_options[item.replace('COMAC ', '').lower()] = 'ignore'
                self.convert_options[item.lower()] = 'ignore'
                self.convert_options[item.lower().replace(' ','')] = 'ignore'
                self.convert_options[item] = 'ignore'
        print(self.convert_options)


    def convert_XML_to_DOTA(self,filename,output_root):
        mydoc = ET.parse(filename)
        root = mydoc.getroot()

        objects = root.find('objects')
        self.items = objects.findall('object')
        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        flag = 0
        ann_list = []
        for item in self.items:
            label = item.find('possibleresult')
            points = item.find('points')
            label=label.find('name').text
            mapped_label = self.convert_options[label] if label in self.convert_options.keys() else self.convert_options[label.lower()]
            if mapped_label == 'ship':
                flag =1
                points = [[int(float(item)) for item in point.text.split(',')] for point in points.findall('point')]
                x1, y1 = points[0]
                x2, y2 = points[1]
                x3, y3 = points[2]
                x4, y4 = points[3]
                ann = [x1, y1, x2, y2, x3, y3, x4, y4, mapped_label, 1]
                for num,val in enumerate(ann[:8]):
                    if val <0:
                        ann[num] = 0
                ann = [str(item) for item in ann]
                ann_list.append(' '.join(ann))
                # print (label, mapped_label, x1, y1, x2, y2, x3, y3, x4, y4)
            else:
                continue
        if flag == 1:
            with open(output_root+"/annfiles/F{:0>4d}.txt".format(int(filename)), 'w') as f:
                f.write('\n'.join(ann_list))
            shutil.copy(os.path.join(images_files, filename+".tif"),
                        os.path.join(output_root+'/images',"F{:0>4d}.jpg".format(int(filename))))

if __name__ ==  '__main__':
    data_root = r"D:\omq\omqdata\rgb\FAIR1M1.0\train"
    images_files = os.path.join(data_root,"images/")
    output_root = r"D:\omq\omqdata\rgb\custom\FAIR1M"
    xml_files = os.listdir(os.path.join(data_root,"annfiles"))
    os.makedirs(output_root+'/annfiles', exist_ok=True)
    os.makedirs(output_root+'/images', exist_ok=True)
    f2d = fair2dota()
    for file in tqdm(xml_files):
        # print(file)
        # if "3661" in file:
        #     print("d")
        # else:
        #     continue
        raster = os.path.join(os.path.join(data_root, "annfiles"), file)
        f2d.convert_XML_to_DOTA(raster,output_root)


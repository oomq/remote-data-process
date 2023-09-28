import shutil
import numpy as np
from PIL import Image,ImageDraw
import os
import os.path as osp
import cv2
import json
from tqdm import tqdm

###用中心点定位裁剪
class Fair2dota():
    def __init__(self, images_path, txt_path, save_path):
        self.images_path = images_path
        self.txt_path = txt_path
        self.save_dir = save_path
        self.save_dir_images = osp.join(save_path, "images")
        self.save_dir_txt = osp.join(save_path, "annfiles")
        os.makedirs(self.save_dir_images, exist_ok=True)
        os.makedirs(self.save_dir_txt, exist_ok=True)
        self.save_dir_super_images = osp.join(save_path, "super-resolution_img")
        self.save_dir_super_ann = osp.join(save_path, "super-resolution_ann")
        os.makedirs(self.save_dir_super_images, exist_ok=True)
        os.makedirs(self.save_dir_super_ann, exist_ok=True)
        self.crop_size_512 = 512
        self.crop_size_1024 = 1024
        self.count = 0
        self.image_names=[]
        self.sum_mat=[]

    def cut_image(self, img):
        pass

    def run(self):
        for filename in tqdm(os.listdir(self.images_path)):
            x1 = x2 = y1 = y2 = 0
            # if "F3427" in filename:
            #     print("d")
            # else:
            #     continue

            img_path = osp.join(self.images_path, filename)
            ann_path = osp.join(txt_path, filename.replace("jpg", "txt"))
            img_ori = cv2.imread(img_path)
            img1= Image.open(img_path)
            width, height = img1.size
            resolution = f'{height}x{width}'

            with open(ann_path, "r") as f:
                data = f.readlines()

            # 创建一个空的矩阵来存储数据
            matrix = []
            # 遍历每一行数据
            for line in data:
                # 将每一行的数据按空格分割，并将前八个数字转换为整数
                values = [int(x) for x in line.split()[:8]]
                # 将这些数字添加到矩阵中
                matrix.append(values)

            # 将矩阵转换为NumPy数组（如果需要）
            matrix = np.array(matrix)
            ###标注会超出图片边界，需要归正
            min_x = max(np.min(matrix[:, [0, 2, 4, 6]]), 0)
            max_x = min(np.max(matrix[:, [0, 2, 4, 6]]), width)
            min_y = max(np.min(matrix[:, [1, 3, 5, 7]]), 0)
            max_y = min(np.max(matrix[:, [1, 3, 5, 7]]), height)

            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            w = max_x - min_x
            h = max_y - min_y


            if width > 2000:
                shutil.copy(img_path, osp.join(self.save_dir_super_images, filename))
                shutil.copy(ann_path, osp.join(self.save_dir_super_ann,filename.replace("jpg", "txt")))

            if width == 2000:
                if w < self.crop_size_1024 and h < self.crop_size_1024:
                    # 调整中心坐标以避免超出图像边界
                    cx = np.clip(cx, self.crop_size_1024 / 2, width - self.crop_size_1024 / 2)
                    cy = np.clip(cy, self.crop_size_1024 / 2, height - self.crop_size_1024 / 2)

                    # 计算新的裁剪框坐标
                    x1 = int(cx - self.crop_size_1024 / 2)
                    y1 = int(cy - self.crop_size_1024 / 2)
                    x2 = x1 + self.crop_size_1024
                    y2 = y1 + self.crop_size_1024

                    # 对矩阵中的每个点进行偏移
                    matrix[:, [0, 2, 4, 6]] -= x1
                    matrix[:, [1, 3, 5, 7]] -= y1

                    ##img_save
                    img = img_ori[y1:y2, x1:x2]
                    # print(x1, x2, y1, y2)
                    if x2 > width or y2 > height or img.shape[0] < self.crop_size_1024 or img.shape[1] < self.crop_size_1024:
                        print("gg")

                    # for points in matrix:
                    #     # 将四个点的坐标提取出来
                    #     x1, y1, x2, y2, x3, y3, x4, y4 = points
                    #
                    #     # 组织点坐标成矩阵
                    #     points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
                    #
                    #     # 在图像上绘制旋转框
                    #     cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                    cv2.imwrite(osp.join(self.save_dir_images, filename), img=img)

                    with open(osp.join(self.save_dir_txt, filename.replace("jpg", "txt")), 'w') as file:
                        for row in matrix:
                            # 将每一行的数值转换为字符串并以空格分隔
                            row_str = ' '.join(map(str, row))
                            # 将行字符串与其他信息组合起来（例如 'ship 1'）
                            line = f"{row_str} ship 1\n"
                            # 将行写入文件
                            file.write(line)
                else:
                    shutil.copy(img_path, osp.join(self.save_dir_super_images, filename))
                    shutil.copy(ann_path, osp.join(self.save_dir_super_ann, filename.replace("jpg", "txt")))
                    self.count += 1


            if width == 1500:
                continue
                if w < self.crop_size_1024 and h < self.crop_size_1024:

                    # 调整中心坐标以避免超出图像边界
                    cx = np.clip(cx, self.crop_size_1024 / 2, width - self.crop_size_1024 / 2)
                    cy = np.clip(cy, self.crop_size_1024 / 2, height - self.crop_size_1024 / 2)

                    # 计算新的裁剪框坐标
                    x1 = int(cx - self.crop_size_1024 / 2)
                    y1 = int(cy - self.crop_size_1024 / 2)
                    x2 = x1 + self.crop_size_1024
                    y2 = y1 + self.crop_size_1024

                    # 对矩阵中的每个点进行偏移
                    matrix[:, [0, 2, 4, 6]] -= x1
                    matrix[:, [1, 3, 5, 7]] -= y1

                    ##img_save
                    img = img_ori[y1:y2, x1:x2]
                    # print(x1, x2, y1, y2)
                    if x2 > width or y2 > height or img.shape[0] < self.crop_size_1024 or img.shape[1] < self.crop_size_1024:
                        print("gg")

                    # for points in matrix:
                    #     # 将四个点的坐标提取出来
                    #     x1, y1, x2, y2, x3, y3, x4, y4 = points
                    #
                    #     # 组织点坐标成矩阵
                    #     points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
                    #
                    #     # 在图像上绘制旋转框
                    #     cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                    cv2.imwrite(osp.join(self.save_dir_images, filename), img=img)

                    with open(osp.join(self.save_dir_txt, filename.replace("jpg", "txt")), 'w') as file:
                        for row in matrix:
                            # 将每一行的数值转换为字符串并以空格分隔
                            row_str = ' '.join(map(str, row))
                            # 将行字符串与其他信息组合起来（例如 'ship 1'）
                            line = f"{row_str} ship 1\n"
                            # 将行写入文件
                            file.write(line)
                else:
                    img_resized =cv2.resize(img_ori,(1024,1024))
                    matrix[:, :] = matrix[:, :]*(1024/1500)
                    # for points in matrix:
                    #     # 将四个点的坐标提取出来
                    #     x1, y1, x2, y2, x3, y3, x4, y4 = points
                    #
                    #     # 组织点坐标成矩阵
                    #     points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
                    #
                    #     # 在图像上绘制旋转框
                    #     cv2.polylines(img_resized, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.imwrite(osp.join(self.save_dir_images, filename), img=img_resized)
                    with open(osp.join(self.save_dir_txt, filename.replace("jpg", "txt")), 'w') as file:
                        for row in matrix:
                            # 将每一行的数值转换为字符串并以空格分隔
                            row_str = ' '.join(map(str, row))
                            # 将行字符串与其他信息组合起来（例如 'ship 1'）
                            line = f"{row_str} ship 1\n"
                            # 将行写入文件
                            file.write(line)

                    self.count += 1


            if width ==1000:
                continue
                # 创建一个800x800的空白画布，背景颜色可以根据需求选择
                canvas = Image.new('RGB', (1024, 1024), (255, 255, 255))

                # 在空白画布上粘贴600x800的图像
                canvas.paste(img1, (0, 0))
                # draw = ImageDraw.Draw(canvas)
                ###vis
                # for points in matrix:
                #     # 将四个点的坐标提取出来
                #     x1, y1, x2, y2, x3, y3, x4, y4 = points
                #
                #     draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline=(255, 0, 0), width=2)
                canvas.save(osp.join(self.save_dir_images,"1000", filename))
                with open(osp.join(self.save_dir_txt,"1000", filename.replace("jpg", "txt")), 'w') as file:
                    for row in matrix:
                        # 将每一行的数值转换为字符串并以空格分隔
                        row_str = ' '.join(map(str, row))
                        # 将行字符串与其他信息组合起来（例如 'ship 1'）
                        line = f"{row_str} ship 1\n"
                        # 将行写入文件
                        file.write(line)
                # 关闭图像对象
                canvas.close()


            if width < 1000: ###512
                continue
                if w < self.crop_size_512 and h < self.crop_size_512:
                    # 调整中心坐标以避免超出图像边界
                    cx = np.clip(cx, self.crop_size_512 / 2, width - self.crop_size_512 / 2)
                    cy = np.clip(cy, self.crop_size_512 / 2, height - self.crop_size_512 / 2)

                    # 计算新的裁剪框坐标
                    x1 = int(cx - self.crop_size_512 / 2)
                    y1 = int(cy - self.crop_size_512 / 2)
                    x2 = x1 + self.crop_size_512
                    y2 = y1 + self.crop_size_512

                    # 对矩阵中的每个点进行偏移
                    matrix[:, [0, 2, 4, 6]] -= x1
                    matrix[:, [1, 3, 5, 7]] -= y1

                    ##img_save
                    img = img_ori[y1:y2, x1:x2]
                    # print(x1, x2, y1, y2)
                    if x2 > width or y2 > height or img.shape[0] < 512 or img.shape[1] < 512:
                        print("gg")

                    # for points in matrix:
                    #     # 将四个点的坐标提取出来
                    #     x1, y1, x2, y2, x3, y3, x4, y4 = points
                    #
                    #     # 组织点坐标成矩阵
                    #     points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
                    #
                    #     # 在图像上绘制旋转框
                    #     cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                    cv2.imwrite(osp.join(self.save_dir_images,"512", filename), img=img)

                    with open(osp.join(self.save_dir_txt,"512", filename.replace("jpg", "txt")), 'w') as file:
                        for row in matrix:
                            # 将每一行的数值转换为字符串并以空格分隔
                            row_str = ' '.join(map(str, row))
                            # 将行字符串与其他信息组合起来（例如 'ship 1'）
                            line = f"{row_str} ship 1\n"
                            # 将行写入文件
                            file.write(line)

                else:
                    # 创建一个800x800的空白画布，背景颜色可以根据需求选择
                    canvas = Image.new('RGB', (800, 800), (255, 255, 255))

                    # 在空白画布上粘贴600x800的图像
                    canvas.paste(img1, (0, 0))
                    canvas = canvas.resize((512, 512))
                    draw = ImageDraw.Draw(canvas)
                    matrix = matrix*(512/800)
                    ###vis
                    # for points in matrix:
                    #     # 将四个点的坐标提取出来
                    #     x1, y1, x2, y2, x3, y3, x4, y4 = points

                        # draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline=(255, 0, 0), width=2)
                    canvas.save(osp.join(self.save_dir_images,"512", filename))
                    with open(osp.join(self.save_dir_txt,"512", filename.replace("jpg", "txt")), 'w') as file:
                        for row in matrix:
                            # 将每一行的数值转换为字符串并以空格分隔
                            row_str = ' '.join(map(str, row))
                            # 将行字符串与其他信息组合起来（例如 'ship 1'）
                            line = f"{row_str} ship 1\n"
                            # 将行写入文件
                            file.write(line)
                    # 关闭图像对象
                    canvas.close()
                img1.close()



                    # _img = img_ori[max_y-min_y,max_x-min_x]
                    # cv2.resize(img_ori,[512,512])

        print(self.count)

    def image_compose(self,i=None):
        to_image = Image.new('RGB', (1024,1024))  # 创建一个新图

        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(1, 3):
            for x in range(1, 3):

                from_image = Image.open(osp.join(self.save_dir_images,"512",
                                                 self.image_names[i + 2 * (y - 1) + (x - 1)]))

                to_image.paste(from_image, ((x - 1) * self.crop_size_512, (y - 1) * self.crop_size_512))
        return to_image

    def ann_compose(self,i=None,f=None):
        for y in range(1, 2+ 1):
            for x in range(1, 2 + 1):
                with open(osp.join(self.save_dir_txt,"512",self.image_names[i+2*(y-1)+(x-1)].
                                         replace("jpg","txt")),encoding='utf-8') as w:
                    data = w.readlines()

                # 创建一个空的矩阵来存储数据
                mat = []
                # 遍历每一行数据
                for line in data:
                    # 将每一行的数据按空格分割，并将前八个数字转换为整数
                    values = [int(round(float(x))) for x in line.split()[:8]]
                    # 将这些数字添加到矩阵中
                    mat.append(values)

                mat = np.array(mat)
                mat[:, [0, 2, 4, 6]] += (x-1)*512
                mat[:, [1, 3, 5, 7]] += (y-1)*512

                for row in mat:
                    # 将每一行的数值转换为字符串并以空格分隔
                    row_str = ' '.join(map(str, row))
                    # 将行字符串与其他信息组合起来（例如 'ship 1'）
                    line = f"{row_str} ship 1\n"
                    # 将行写入文件
                    f.write(line)

    def compose(self):
        img_512_path = osp.join(self.save_dir_images,"512")
        self.image_names = [img for img in os.listdir(img_512_path)]
        for i in tqdm(range(0, len(self.image_names), 4)):
            img = self.image_compose(i=i)  # 调用函数
            name = osp.join(self.save_dir_images, "FA{:0=4d}.jpg".format(i))

            with open(osp.join(self.save_dir_txt, "FA{:0=4d}.txt".format(i)),"w+") as f:
                self.ann_compose(i,f)

            img.save(name)







if __name__ == '__main__':
    images_path = r'D:\omq\omqdata\rgb\custom\FAIR1M\images'
    txt_path = r'D:\omq\omqdata\rgb\custom\FAIR1M\annfiles'
    save_path = r"D:\omq\omqdata\rgb\custom\Fair\train"

    r = Fair2dota(images_path, txt_path, save_path)
    r.run()
    # r.compose()



"""
用xyxy定位裁剪框

import shutil
import numpy as np
from PIL import Image
import os
import os.path as osp
import cv2
import json
from tqdm import tqdm


class Fair2dota():
    def __init__(self, images_path, txt_path, save_path):
        self.images_path = images_path
        self.txt_path = txt_path
        self.save_dir = save_path
        self.save_dir_images = osp.join(save_path, "images")
        self.save_dir_txt = osp.join(save_path, "annfiles")
        os.makedirs(self.save_dir_images, exist_ok=True)
        os.makedirs(self.save_dir_txt, exist_ok=True)
        self.save_dir_super_images = osp.join(save_path, "super-resolution")
        os.makedirs(self.save_dir_super_images, exist_ok=True)
        self.crop_size_512 = 512
        self.count =0

    def cut_image(self, img):
        pass

    def run(self):
        for filename in tqdm(os.listdir(self.images_path)):
            x1 = x2 = y1 = y2 = 0
            # if "F3427" in filename:
            #     print("d")
            # else:
            #     continue

            img_path = osp.join(self.images_path, filename)
            img_ori = cv2.imread(img_path)
            with Image.open(img_path) as img1:
                width,height = img1.size
                resolution = f'{height}x{width}'

            if width > 3000:
                shutil.copy(img_path, osp.join(self.save_dir_super_images, filename))
                continue

            if width < 1000:
                with open(osp.join(txt_path, filename.replace("jpg", "txt")), "r") as f:
                    data = f.readlines()

                # 创建一个空的矩阵来存储数据
                matrix = []
                # 遍历每一行数据
                for line in data:
                    # 将每一行的数据按空格分割，并将前八个数字转换为整数
                    values = [int(x) for x in line.split()[:8]]
                    # 将这些数字添加到矩阵中
                    matrix.append(values)

                # 将矩阵转换为NumPy数组（如果需要）
                matrix = np.array(matrix)
                ###标注会超出图片边界，需要归正
                min_x = max(np.min(matrix[:, [0, 2, 4, 6]]), 0)
                max_x = min(np.max(matrix[:, [0, 2, 4, 6]]), width)
                min_y = max(np.min(matrix[:, [1, 3, 5, 7]]), 0)
                max_y = min(np.max(matrix[:, [1, 3, 5, 7]]), height)

                if max_x - min_x < 512 and max_y - min_y < 512:
                    # print(filename, resolution)
                    x1 = min_x
                    y1 = min_y
                    if x1 + self.crop_size_512 > width:
                        x1 = max(max_x - self.crop_size_512, 0)
                    if y1 + self.crop_size_512 > height:
                        y1 = max(max_y - self.crop_size_512, 0)

                    x2 = x1 + self.crop_size_512
                    y2 = y1 + self.crop_size_512

                    # 对矩阵中的每个点进行偏移
                    matrix[:, [0, 2, 4, 6]] -= x1
                    matrix[:, [1, 3, 5, 7]] -= y1

                    ##img_save
                    img = img_ori[y1:y2,x1:x2]
                    # print(x1, x2, y1, y2)
                    if x2 > width or y2 > height or img.shape[0]<512 or img.shape[1]<512 :
                        print("gg")
                    cv2.imwrite(osp.join(self.save_dir_images, filename), img=img)
                    with open(osp.join(self.save_dir_txt, filename.replace("jpg", "txt")), 'w') as file:
                        for row in matrix:
                            # 将每一行的数值转换为字符串并以空格分隔
                            row_str = ' '.join(map(str, row))
                            # 将行字符串与其他信息组合起来（例如 'ship 1'）
                            line = f"{row_str} ship 1\n"
                            # 将行写入文件
                            file.write(line)
                else:
                    pass
                    # _img = img_ori[max_y-min_y,max_x-min_x]
                    # cv2.resize(img_ori,[512,512])
            else:
                self.count+=1
        print(self.count)




if __name__ == '__main__':
    images_path = r'D:\omq\omqdata\rgb\custom\FAIR1M\images'
    txt_path = r'D:\omq\omqdata\rgb\custom\FAIR1M\annfiles'
    save_path = r"D:\omq\omqdata\rgb\custom\Fair"

    r = Fair2dota(images_path, txt_path, save_path)
    r.run()
"""

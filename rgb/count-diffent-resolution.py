import os
from PIL import Image
from tqdm import tqdm
# 设置图像文件夹路径
# image_folder = r'D:\omq\omqdata\rgb\custom\FAIR1M\images'
image_folder = r'D:\omq\omqdata\rgb\TGRS-HRRSD-Dataset-mstr-gthb\OPT2017\JPEGImages'

# image_folder =r"D:\omq\omqdata\rgb\HRSC2016\HRSC2016\Train\AllImages"
# 创建一个字典来存储不同分辨率的图像数量
resolution_count = {}

# 遍历文件夹中的所有图像文件
max_width,max_height =0,0
for filename in tqdm(os.listdir(image_folder)):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif',".bmp")):  # 检查文件扩展名
        image_path = os.path.join(image_folder, filename)

        # 打开图像并获取其分辨率
        with Image.open(image_path) as img:
            width, height = img.size
            resolution = f'{width}x{height}'


            # 更新分辨率计数字典
            if resolution in resolution_count:
                resolution_count[resolution] += 1
            else:
                resolution_count[resolution] = 1

# 打印不同分辨率的图像数量
for resolution, count in resolution_count.items():
    print(f'Resolution {resolution}: {count} images')
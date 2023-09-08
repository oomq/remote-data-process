import os

import cv2
import numpy as np

root = r"example/"
save_path = r"blur/"
os.makedirs(save_path, exist_ok=True)

for img in os.listdir(root):
    image = cv2.imread(root+img)

    # image = cv2.dilate(image, (5,5), iterations=2)
    image = cv2.erode(image, (3,3), iterations=1)#腐蚀

    # image = cv2.Laplacian(image, cv2.CV_64F)

    image = cv2.blur(image,(1,1))
    # dilated_image = cv2.dilate(image, (7,7), iterations=2)#膨胀
    # image = cv2.erode(image, (3,3), iterations=1)
    # 分离RGB通道
    b, g, r = cv2.split(image)

    # 计算每个通道的标准差
    std_dev_r = np.std(r)
    std_dev_g = np.std(g)
    std_dev_b = np.std(b)
    print(img,std_dev_r)
    # 设置拉伸的参数（通常为1-2倍标准差）
    stretch_factor = 300.0

    # 计算拉伸后的通道值
    stretched_r = (r - np.mean(r)) / std_dev_r * stretch_factor + np.mean(r)
    stretched_g = (g - np.mean(g)) / std_dev_g * stretch_factor + np.mean(g)
    stretched_b = (b - np.mean(b)) / std_dev_b * stretch_factor + np.mean(b)

    # 合并拉伸后的通道
    stretched_image = cv2.merge((stretched_b, stretched_g, stretched_r))

    # stretched_image = cv2.blur(stretched_image,(3,3))

    # 保存结果图像
    cv2.imwrite(save_path+img, stretched_image)
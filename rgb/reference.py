# python批量更换后缀名
import os
import sys
#需要修改后缀的文件目录
os.chdir(r'D:\omqdata\sar\dota\SRSDD\test\images')

# 列出当前目录下所有的文件
files = os.listdir('./')
print('files',files)

for fileName in files:
    portion = os.path.splitext(fileName)
    newName = portion[0] + ".jpg" #修改为目标后缀
    os.rename(fileName, newName)
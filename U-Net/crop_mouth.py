# -*-coding:utf-8-*-
from PIL import Image
import os

img_dir = 'F:/pre_data/CelebA/Img/'
mouth_dir = 'F:/pre_data/CelebA/mouth/'
img_list = os.listdir(img_dir)

x ,y ,w ,h= 25 ,60 ,128, 128

for i in img_list:
    im = Image.open(os.path.join(img_dir,i))
    # 图片的宽度和高度
    img_size = im.size
    #print("图片宽度和高度分别是{}".format(img_size))
    '''
    裁剪：传入一个元组作为参数
    元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
    '''
    # 截取图片中区域为56*32
    region = im.crop((x, y, x+w, y+h))
    region.save(os.path.join(mouth_dir,i))

# for j in img_list:
#     im = Image.open(os.path.join(mouth_dir,i))
#     clrs = im.getcolors()
#     print(clrs)
#     # if len(cls) == 2:
#     #     os.remove(os.path.join(mouth_dir,i))
    
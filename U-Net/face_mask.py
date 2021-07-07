# -*-coding:utf-8-*-
from PIL import Image
import os
import random
from skimage import io,data
import numpy as np
import cv2

img_dir = 'F:/pre_data/CelebA/face'
flist = os.listdir(img_dir)
save_dir = 'F:/pre_data/CelebA/mask_mouth/'
#save_dir = 'F:/pre_data/CelebA/mask_eyes/'
#save_dir = 'F:/pre_data/CelebA/mask_nose/'
#mouth
x,y,w,h = 38,85,56,32
id_x = np.arange(85,118)
id_y = np.arange(38,95)
# #nose
# x,y,w,h=52,38,24,52
# id_x = np.arange(38,91)
# id_y = np.arange(52,77)
# #eyes
# x,y,w,h=34,38,64,28
# id_x = np.arange(38,67)
# id_y = np.arange(34,99)

mask = np.zeros((128,128,3))
for idx in range(0,128):
    for idy in range(0,128):
        if idx in id_x and idy in id_y:
            mask[idx, idy][0] = 1
            mask[idx, idy][1] = 1
            mask[idx, idy][2] = 1
for i in flist:
    image = cv2.imread(os.path.join(img_dir,str(i)))
    mask_image = mask * image
    print(mask_image.shape)
    cv2.imwrite(save_dir+i,mask_image)

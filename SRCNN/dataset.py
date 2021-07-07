import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
     
    
def load_video_image(file_path, output_height=None, output_width=None,
              is_gray=False, scale=1.0, is_scale_back=False):
    
    if output_width is None:
        output_width = output_heigh
    
    img = Image.open(file_path)
    img_lr = Image.open(file_path)
    if is_gray is False and img.mode is not 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode is not 'L':
        img = img.convert('L')
      
    #print(scale)
    img = img.resize((output_width, output_height),Image.BICUBIC)
    img_lr = img.resize((int(output_width//scale),int(output_height//scale)),Image.BICUBIC)
    img_lr = img_lr.resize((output_width, output_height),Image.BICUBIC), img
    else:
        return img_lr, img
      
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_pathoutput_height=128, output_width=None,
              is_gray=False, upscale=4.0):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list        
        self.upscale = upscale
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.is_gray = is_gray
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
    
        if self.is_mirror:
            is_mirror = random.randint(0,1) is 0
        else:
            is_mirror = False
          
        lr, hr = load_video_image(join(self.root_path, self.image_filenames[index]), 
                                  self.output_height, self.output_width,
                                  self.is_gray, self.upscale)

        lr = np.array(lr, dtype=np.float32) / 255.0
        hr = np.array(hr, dtype=np.float32) / 255.0

        # noise = np.random.normal(0,25/255.0,lr.shape)
        # lr = lr + noise

        input = self.input_transform(lr)
        target = self.input_transform(hr)
        input = input.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        
        return input, target

    def __len__(self):
        return len(self.image_filenames)

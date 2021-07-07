import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import numpy as np
            
def load_video_image(data_path, output_height=128, output_width=128,scale=4.0):
    
    img_hr = Image.open(data_path)
    #print(scale)
    img_lr = img_hr.resize((output_width//scale,output_height//scale),Image.BICUBIC)
    img_lr = img_lr.resize((output_width,output_height),Image.BICUBIC)
    
    return img_lr, img_hr
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, data_root,output_height=128, output_width=None,upscale=4.0):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list        
        self.upscale = upscale
        self.output_height = output_height
        self.output_width = output_width
        self.data_root = data_root
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
    
        lr, hr = load_video_image(join(self.data_root, self.image_filenames[index]), 
                                  self.output_height, self.output_width,self.upscale)
       
        lr = np.array(lr,dtype=np.float32)/255.0
        hr=np.array(hr,dtype=np.float32)/255.0
        #noise = np.random.normal(0,25/255.0,lr.shape)
        #lr = lr+noise
        input = self.input_transform(lr)
        target = self.input_transform(hr)
        input=input.type(torch.FloatTensor)
        target=target.type(torch.FloatTensor)        

        return input, target

    def __len__(self):
        return len(self.image_filenames)
        
        
        

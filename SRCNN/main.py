from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from model import *
from math import log10
import torchvision
import cv2
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--nrow', type=int, help='number of the rows to save images', default=1)
parser.add_argument('--dataroot', default="data/train", type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--save_iter', type=int, default=10, help='the interval iterations for saving models')
parser.add_argument('--test_iter', type=int, default=300, help='the interval iterations for testing')
parser.add_argument('--output_height', type=int, default=128, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=128, help='the width  of the output image to network')
parser.add_argument('--upscale', type=int, default=2, help='the depth of wavelet tranform')
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='results/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

loss_data = open('loss.csv','a',newline='')
csv_write = csv.writer(loss_data,dialect='excel')
name_list = ['epoch','loss_img']
csv_write.writerow(name_list)


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    ngpu = int(opt.ngpu)  
    #--------------build models--------------------------
    srnet = SRCNN()
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            pretrained_dict = weights['model'].state_dict()
            model_dict = srnet.state_dict()
            # print(model_dict)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            srnet.load_state_dict(model_dict)
            # srnet.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    print(srnet)
     
    criterion_r = nn.KLDivLoss(size_average=True)
    criterion_m = nn.MSELoss(size_average=True)
    
    if opt.cuda:
        srnet = srnet.cuda()
        criterion_r = criterion_r.cuda()
        criterion_m = criterion_m.cuda()
    
    #-----------------load dataset--------------------------
    train_list = os.listdir(opt.dataroot)
    train_set = ImageDatasetFromFile(train_list, opt.dataroot, 
              output_height=opt.output_height, output_width=opt.output_width,
              is_gray=False, upscale=opt.scale)    
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))
    
    start_time = time.time()
    srnet.train()
    #----------------Train by epochs--------------------------
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        if epoch%opt.save_iter == 0:
            save_checkpoint(srnet, epoch, 0, 'sr_')
            
        optimizer_sr = optim.Adam(srnet.parameters(), lr=opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0005)
        #optimizer_res = optim.Adam(res_conv.parameters(),lr=opt.lr,betas=(opt.momentum,0.999),weight_decay=0.0005)
        
        for iteration, batch in enumerate(train_data_loader, 0):
            #--------------train------------
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
              
            prediction = forward_parallel(srnet, input, opt.ngpu)            

            loss_img = criterion_m(prediction,target)
         
            optimizer_sr.zero_grad()    
            loss_img.backward()                       
            optimizer_sr.step()
            
            info = "===> Epoch[{}]({}/{}): time: {:4.4f}:".format(epoch, iteration, len(train_data_loader), time.time()-start_time)
            info += "Loss_img: {:.4f}".format(loss_img.data)            
                          
            print(info)

        if epoch%2 ==0:
            #loss_list:0 epoch;1 loss_lr;2 loss_hr;3 loss_textures;4 loss_img;5 loss
            loss_list =['{}'.format(epoch),'{}'.format(loss_img)] 
            csv_write.writerow(loss_list)

def lr_schedule(epoch):
    initial_lr = opt.lr
    if epoch <= 50:
        lr = initial_lr
    elif epoch <= 80:
        lr = initial_lr / 10
    elif epoch <= 120:
        lr = initial_lr / 100
    else:
        lr = lr / 200
    #log('current learning rate is %2.8f' % lr)
    return lr

def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)
            
def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "model/" + prefix +"model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def save_images(images, name, path, nrow=10):
    #print(images.size())
    img = images.cpu()
    im = img.data.numpy().astype(np.float32)
    #print(im.shape)
    im = im.transpose(0,2,3,1)
    imsave(im, [nrow, int(math.ceil(im.shape[0]/float(nrow)))], os.path.join(path, name) )

def merge(images, size):
    #print(images.shape())
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    #print(img)
    for idx, image in enumerate(images):
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    img = merge(images, size)
    # print(img)
    return cv2.imwrite(path, img)

if __name__ == "__main__":
    main()    

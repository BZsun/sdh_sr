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
from networks import *
from math import log10
import torchvision
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


parser = argparse.ArgumentParser()
parser.add_argument('--testroot', default="data/celeba_test", type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--test_batchSize', type=int, default=1, help='test batch size')
parser.add_argument('--output_height', type=int, default=128, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=128, help='the width  of the output image to network')
parser.add_argument('--upscale', type=int, default=4, help='the depth of wavelet tranform')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='mouth_sr/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained", default="model/sr_model_epoch_200_iter_0.pth", type=str, help="path to pretrained model (default: none)")

opt = parser.parse_args()

def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)

def save_images(images, name, path, nrow=10):
    # print(images.size())
    img = images.cpu()
    im = img.data.numpy().astype(np.float32)
    # print(im.shape)
    im = im.transpose(0, 2, 3, 1)
    imsave(im, [nrow, int(math.ceil(im.shape[0] // float(nrow)))], os.path.join(path, name))


def merge(images, size):
    # print(images.shape())
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    # print(img)
    for idx, image in enumerate(images):
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def imsave(images, size, path):
    img = merge(images, size)
    # print(img)
    return cv2.imwrite(path,img)

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

# --------------build models--------------------------
srnet = RCAN()

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

if opt.cuda:
    srnet = srnet.cuda()

#-----------------load dataset--------------------------
test_list = os.listdir(opt.testroot)
test_set = ImageDatasetFromFile(test_list, opt.testroot,output_height=opt.output_height, 
                                        output_width=opt.output_width,upscale=opt.upscale)
test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batchSize,
                                        shuffle=False, num_workers=int(opt.workers))
srnet.eval()
avg_psnr = 0
avg_ssim = 0

for titer, batch in enumerate(test_data_loader, 0):
    input, target = Variable(batch[0]), Variable(batch[1])
    if opt.cuda:
        input = input.cuda()
        target = target.cuda()

    prediction = forward_parallel(srnet, input, opt.ngpu)
    prediction_s = prediction.cpu().detach().numpy().squeeze(0).transpose((1,2,0))
    target_s = target.cpu().detach().numpy().squeeze(0).transpose((1,2,0))
    psnr = compare_psnr(im_test=prediction_s,im_true=target_s,data_range=None)  
    ssim = compare_ssim(prediction_s, target_s,multichannel=True)
    
    avg_psnr += psnr
    avg_ssim += ssim
    save_images(prediction, "test_{:02d}.jpg".format(titer),path=opt.outf,nrow=1)
    
    print("test_{:02d}.jpg psnr : {:.4f} dB".format(titer,psnr))
    print("test_{:02d}.jpg ssim : {:.4f}".format(titer,ssim) )

print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))
print("===> Avg. SSIM: {:.4f}".format(avg_ssim / len(test_data_loader)))

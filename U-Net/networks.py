import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import pdb
import torch.nn.functional as F

def loss_MSE(x, y, size_average=False):
    z = x - y
    z2 = z * z
    if size_average:
        return z2.mean()
    else:
        return z2.sum().div(x.size(0)*2)
    
def loss_L1(x, y, size_average=False):
    z = abs(x - y)
    if size_average:
        return z.mean()
    else:
        return z.sum().div(x.size(0)*2)
    
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
    xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
    yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
    xi2 = torch.sum(xi * xi, dim=2)
    yi2 = torch.sum(yi * yi, dim=2)
  
    out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
    return torch.mean(out)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, inc, outc):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1,groups=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1,groups=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, inc, outc):
        super(inconv, self).__init__()
        self.conv = double_conv(inc, outc)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, inc, outc):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(inc, outc)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, inc, outc, bilinear=True):
        super(up, self).__init__()  

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(inc//2, inc//2, 2, stride=2)

        self.up = nn.ConvTranspose2d(inc//2, inc//2, 2, stride=2)
        self.conv = double_conv(inc, outc)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, inc, outc):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, 
                                        stride=1, padding=0,groups=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(UNet, self).__init__()
        self.inc = inconv(inc, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, outc)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        out = self.outc(x9)
             
        return out


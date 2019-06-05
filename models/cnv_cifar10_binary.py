# Source: https://github.com/Xilinx/BNN-PYNQ/blob/master/bnn/src/training/cnv.py

# CNV architecture:
    # 0- baseline coffe code: https://github.com/Xilinx/FINN/blob/master/FINN/inputs/cnv-w1a1.prototxt
    # 1- inputs 24bits (8 bit per channel)
    # 2- there is no Hardtranh block
    # 3- batch normalization is using momentum=0.05 --> nn.BatchNorm2d(64*self.infl_ratio, momentum=0.95),
    #       http://graphics.cs.cmu.edu/courses/16-824/2016_spring/slides/caffe_tutorial.pdf
    # 4- normalization: https://discuss.pytorch.org/t/understanding-transform-normalize/21730

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

class CNV_Cifar10_Binary(nn.Module):

    def __init__(self, num_classes=1000):
        super(CNV_Cifar10_Binary, self).__init__()
        self.infl_ratio=1;

        self.features = nn.Sequential(
            BinarizeConv2d(3, 64*self.infl_ratio, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(64*self.infl_ratio, 64*self.infl_ratio, kernel_size=3, padding=0, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(64*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=0, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256*self.infl_ratio, 256, kernel_size=3, padding=0, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True)
        )

        self.classifier = nn.Sequential(
            BinarizeLinear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(512, num_classes, bias=False),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )
        self.ramin_tester = nn.Sequential(
            BinarizeConv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        #print(x.max())
        #print(x.min())
        #print(x[:,0,0,0])
        #x = self.ramin_tester(x)
        #print(x[:,0,0,0])
        #exit()

        x = self.features(x)
        x = x.view(-1, 256)
        x = self.classifier(x)
        return x

def cnv_cifar10_binary(**kwargs):
    num_classes = getattr(kwargs,'num_classes', 10)
    return CNV_Cifar10_Binary(num_classes)

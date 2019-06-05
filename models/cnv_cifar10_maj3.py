import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
from .majority3_cuda import * 

class CNV_Cifar10_Maj3(nn.Module):

    def __init__(self, num_classes=1000, backprop='majority', padding=1):
        super(CNV_Cifar10_Maj3, self).__init__()
        self.infl_ratio=1;
        self.padding=padding

        self.features = nn.Sequential(
            BinarizeConv2d(3, 64*self.infl_ratio, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            Maj3(64*self.infl_ratio, 64*self.infl_ratio, kernel_size=3, bias=False, backprop=backprop, padding=padding),
            #BinarizeConv2d(64*self.infl_ratio, 64*self.infl_ratio, kernel_size=3, padding=padding, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            Maj3(64*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, bias=False, backprop=backprop, padding=padding),
            #BinarizeConv2d(64*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=padding, bias=True),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            Maj3(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, bias=False, backprop=backprop, padding=padding),
            #BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=padding, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            Maj3(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, bias=False, backprop=backprop, padding=padding),
            #BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=padding, bias=True),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            Maj3(256*self.infl_ratio, 256, kernel_size=3, bias=False, backprop=backprop, padding=padding),
            #BinarizeConv2d(256*self.infl_ratio, 256, kernel_size=3, padding=padding, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(256 * 4 * 4, 512, bias=False),
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

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.classifier(x)
        return x

def cnv_cifar10_maj3(**kwargs):
    num_classes = getattr(kwargs,'num_classes', 10)
    backprop = kwargs.get('backprop')
    return CNV_Cifar10_Maj3(num_classes, backprop)


import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
from .majority_cuda import * 

__all__ = ['efficientnet_binary']

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bntan=True, majority="B", backprop='normalConv'):
        super(BasicBlock, self).__init__()

        if (majority == "M"):
            self.conv1 = MajConv(inplanes, planes, kernel_size=3, backprop=backprop, padding=1)
            if (stride > 1):
                self.ds = lambda x: torch.nn.functional.interpolate(x, scale_factor=(0.5,0.5))
            else:
                self.ds = lambda x: x
        else:
            self.conv1 = Binaryconv3x3(inplanes, planes, stride)
            self.ds = lambda x: x

        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        
        # the second Convolution is not replaced
        # if you want to replace it, you can use the folowing conditional state
        #if (majority == "M"):
        #    self.conv2 = MajConv(planes, planes, kernel_size=3, backprop=backprop, padding=1)
        #else:
        #    self.conv2 = Binaryconv3x3(planes, planes)

        self.conv2 = Binaryconv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, x):

        residual = x.clone()

        out = self.conv1(x)
        out = self.ds(out)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        import pdb; pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out

class MobileNetBlock(nn.Module):
    def __init__(self, d_in, ch_in, ch_out, expansion=1, stride=1, SE_en=True, majority="B", backprop='normalConvs'):
        super(MobileNetBlock, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        self.SE_en = SE_en

        # SE paper mentioned 16 but for Efficient net it should be 8
        self.SE_rate = 8

        self.conv1 = BinarizeConv2d(ch_in, ch_in*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_in*expansion)
        self.tanh1 = nn.Hardtanh(inplace=True)

        self.conv2 = BinarizeConv2d(ch_in*expansion, ch_in*expansion, kernel_size=3, stride=stride, padding=1, bias=False, groups=ch_in*expansion)
        self.bn2 = nn.BatchNorm2d(ch_in*expansion)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.conv3 = BinarizeConv2d(ch_in*expansion, ch_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch_out)
        self.tanh3 = nn.Hardtanh(inplace=True)

        if ((stride == 1) and (self.ch_in == self.ch_out)):
        	self.bn_res = nn.BatchNorm2d(ch_out)
        	self.tanh_res = nn.Hardtanh(inplace=True)

        if SE_en:
	        self.avgpool_se1 = nn.AvgPool2d(int(d_in/stride))
	        self.bn_se1 = nn.BatchNorm2d(ch_out)
	        self.tanh_se1 = nn.Hardtanh(inplace=True)

	        self.fc_se2 = BinarizeLinear(ch_out, int(ch_out/self.SE_rate))
	        self.bn_se2 = nn.BatchNorm1d(int(ch_out/self.SE_rate))
	        self.tanh_se2 = nn.Hardtanh(inplace=True)

	        self.fc_se3 = BinarizeLinear(int(ch_out/self.SE_rate), ch_out)
	        self.bn_se3 = nn.BatchNorm1d(ch_out)
	        self.sigmoid_se3 = nn.Sigmoid()    

    def forward(self, x):
        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.tanh3(out)

        if self.SE_en:        	
        	se_path = self.avgpool_se1(out)
	        se_path = self.bn_se1(se_path)
	        se_path = self.tanh_se1(se_path)
	        

	        se_path = se_path.view(se_path.size(0), -1)
	        se_path = self.fc_se2(se_path)
	        se_path = self.bn_se2(se_path)
	        se_path = self.tanh_se2(se_path)
	        
	        se_path = self.fc_se3(se_path)
	        se_path = self.bn_se3(se_path)
	        se_path = self.sigmoid_se3(se_path)

	        out = torch.einsum('bjik,bj->bjik', out, se_path)

        if ((self.stride == 1) and (self.ch_in == self.ch_out)):
        	#https://github.com/tensorflow/models/blob/1d057dfc32f515a63ab1e23fd72052ab2a954952/research/slim/nets/mobilenet/conv_blocks.py#L163
        	# according to the link if strid is 1 and the input output dimntions are equal we have residual path
        	out += residual
        	out = self.bn_res(out)
        	out = self.tanh_res(out)

        return out


class EfficientNet(nn.Module):

    def __init__(self):
        super(EfficientNet, self).__init__()

    def _make_layer(self, d_in, ch_in, ch_out, repeat, kernel=3, expansion=1, stride=1, majority="B", backprop='normalConvs', SE_en=True):
        
        layers = []    							
        layers.append(MobileNetBlock(d_in, ch_in, ch_out, expansion=expansion, stride=stride, SE_en=SE_en, majority=majority, backprop=backprop))

        for i in range(1, repeat):
            layers.append(MobileNetBlock(int(d_in/stride), ch_out, ch_out, expansion=expansion, stride=1, SE_en=SE_en, majority=majority, backprop=backprop))

        return nn.Sequential(*layers)

    def forward(self, x):
    	x = self.conv1(x)
    	x = self.bn1(x)
    	x = self.tanh1(x)

    	x = self.layer0(x)
    	x = self.layer1(x)
    	x = self.layer2(x)
    	x = self.layer3(x)
    	x = self.layer4(x)
    	x = self.layer5(x)
    	x = self.layer6(x)

    	x = self.conv2(x)
    	x = self.bn2(x)
    	x = self.tanh2(x)

    	x = self.avgpool(x)
    	x = x.view(x.size(0), -1)
    	x = self.bn3(x)
    	x = self.tanh3(x)

    	x = self.fc(x)
    	x = self.bn4(x)
    	x = self.logsoftmax(x)

    	return x

class EfficientNet_imagenet(EfficientNet):
    def __init__(self, num_classes=1000, scale=1,  majority="BBBBBBB", backprop='normalConv', optimizer='Adam'):
        super(EfficientNet_imagenet, self).__init__()
        
        d_ins = [112, 112, 56, 28, 28, 14, 7]
        channels = [16, 24, 40, 80, 112, 192, 320]
        layers = [1, 2, 2, 3, 3, 4, 1]
        expansions = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 1, 2, 2, 1]
        kernels = [3, 3, 5, 3, 5, 5, 3]


        self.conv1 = BinarizeConv2d(3, 32*scale, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32*scale)
        self.tanh1 = nn.Hardtanh(inplace=True)

        self.layer0 = self._make_layer(d_ins[0], 32*scale, channels[0]*scale, layers[0], kernel=kernels[0], expansion=expansions[0], stride=strides[0], majority=majority[0], backprop=backprop)
        self.layer1 = self._make_layer(d_ins[1], channels[0]*scale, channels[1]*scale, layers[1], kernel=kernels[1], expansion=expansions[1], stride=strides[1], majority=majority[1], backprop=backprop)
        self.layer2 = self._make_layer(d_ins[2], channels[1]*scale, channels[2]*scale, layers[2], kernel=kernels[2], expansion=expansions[2], stride=strides[2], majority=majority[2], backprop=backprop)
        self.layer3 = self._make_layer(d_ins[3], channels[2]*scale, channels[3]*scale, layers[3], kernel=kernels[3], expansion=expansions[3], stride=strides[3], majority=majority[3], backprop=backprop)
        self.layer4 = self._make_layer(d_ins[4], channels[3]*scale, channels[4]*scale, layers[4], kernel=kernels[4], expansion=expansions[4], stride=strides[4], majority=majority[4], backprop=backprop)
        self.layer5 = self._make_layer(d_ins[5], channels[4]*scale, channels[5]*scale, layers[5], kernel=kernels[5], expansion=expansions[5], stride=strides[5], majority=majority[5], backprop=backprop)
        self.layer6 = self._make_layer(d_ins[6], channels[5]*scale, channels[6]*scale, layers[6], kernel=kernels[6], expansion=expansions[6], stride=strides[6], majority=majority[6], backprop=backprop)
        
        self.conv2 = BinarizeConv2d(channels[6]*scale, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.avgpool = nn.AvgPool2d(7)
        self.bn3 = nn.BatchNorm1d(1280)
        self.tanh3 = nn.Hardtanh(inplace=True)

        self.fc = BinarizeLinear(1280, num_classes)
        self.bn4 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        init_model(self)

        if (optimizer == 'Adam'):
            self.regime = {
                0: {'optimizer': 'Adam', 'lr': 5e-3},
                101: {'lr': 1e-3},
                142: {'lr': 5e-4},
                184: {'lr': 1e-4},
                220: {'lr': 1e-5}
            }
        elif (optimizer == 'SGD'):
            self.regime = {
                0: {'optimizer': 'SGD', 'lr': 1e-1, 'weight_decay': 1e-4, 'momentum': 0.9},
                81: {'lr': 1e-4},
                122: {'lr': 1e-5, 'weight_decay': 0},
                164: {'lr': 1e-6}
            }
        else:
            raise Exception('The defined training optimizer is not defined. Please use Adam or SGD rather than: {}'.format(optimizer))


def efficientnet_binary(**kwargs):
    num_classes, dataset, majority, backprop, efficientnet_scale = map(kwargs.get, ['num_classes', 'dataset', 'majority', 'backprop', 'efficientnet_scale'])

    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        return EfficientNet_imagenet(num_classes=num_classes, scale=efficientnet_scale, majority=majority, backprop=backprop)

    else:
        raise Exception('efficientnet_binary is not ready for anydataset rather than imagenet')

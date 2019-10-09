# source:
#		https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
#		https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py (official - Author)
#		https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/main.py
#		
#		https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py	(unofficial but so clear)		
#		 
#		https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py   (unofficial)
#
#		https://github.com/zsef123/EfficientNets-PyTorch/blob/master/models/effnet.py (unofficial)


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


######################################
### This section should be checked especially: 
###			- residual path 
###			- using BN and tanh layers
###			- SE_rate which is proposed 16 but it seems that they are using SE_rate=1 --> parameter counting proves it
######################################
class MobileNetBlock(nn.Module):
	def __init__(self, d_in, ch_in, ch_out, kernel=3, expansion=1, stride=1, SE_en=True, majority="B", backprop='normalConvs', dropout_rate=0.2):
		super(MobileNetBlock, self).__init__()

		self.ch_in = ch_in
		self.ch_out = ch_out
		self.stride = stride
		self.SE_en = SE_en
		self.expansion = expansion
		ch_in_expanded = ch_in * expansion

		SE_rate = 0.25

		if expansion != 1:
			self.conv1 = BinarizeConv2d(ch_in, ch_in_expanded, kernel_size=1, bias=False)
			self.bn1 = nn.BatchNorm2d(ch_in_expanded, eps=1e-03, momentum=0.01)
			self.tanh1 = nn.Hardtanh(inplace=True)

		padding = int(( kernel - 1 ) / 2)
		self.conv2 = BinarizeConv2d(ch_in_expanded, ch_in_expanded, kernel_size=kernel, stride=stride, padding=padding, bias=False, groups=ch_in_expanded)
		self.bn2 = nn.BatchNorm2d(ch_in_expanded, eps=1e-03, momentum=0.01)
		self.tanh2 = nn.Hardtanh(inplace=True)

		if SE_en:
			self.avgpool_se1 = nn.AvgPool2d(int(d_in/stride))
			self.bn_se1 = nn.BatchNorm2d(ch_in_expanded, eps=1e-03, momentum=0.01)
			self.tanh_se1 = nn.Hardtanh(inplace=True)

			#self.fc_se2 = BinarizeLinear(ch_out, int(ch_out*SE_rate))
			self.fc_se2 = BinarizeLinear(ch_in_expanded, int(ch_in*SE_rate))
			self.bn_se2 = nn.BatchNorm1d(int(ch_in*SE_rate), eps=1e-03, momentum=0.01)
			self.tanh_se2 = nn.Hardtanh(inplace=True)

			self.fc_se3 = BinarizeLinear(int(ch_in*SE_rate), ch_in_expanded)
			self.bn_se3 = nn.BatchNorm1d(ch_in_expanded, eps=1e-03, momentum=0.01)
			self.sigmoid_se3 = nn.Sigmoid()   

		self.conv3 = BinarizeConv2d(ch_in_expanded, ch_out, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(ch_out, eps=1e-03, momentum=0.01)
		#self.tanh3 = nn.Hardtanh(inplace=True)

		if ((stride == 1) and (self.ch_in == self.ch_out)):
			self.dropout_res = nn.Dropout2d(p=dropout_rate, inplace=False)
			self.bn_res = nn.BatchNorm2d(ch_out, eps=1e-03, momentum=0.01)
		
		self.tanh_res = nn.Hardtanh(inplace=True)

	def forward(self, x):
		residual = x.clone()

		if self.expansion != 1:
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.tanh1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.tanh2(x)

		if self.SE_en:   
			se_path = self.avgpool_se1(x)
			se_path = self.bn_se1(se_path)
			se_path = self.tanh_se1(se_path)
			
			se_path = se_path.view(se_path.size(0), -1)
			se_path = self.fc_se2(se_path)
			se_path = self.bn_se2(se_path)
			se_path = self.tanh_se2(se_path)
			
			se_path = self.fc_se3(se_path)
			se_path = self.bn_se3(se_path)
			se_path = self.sigmoid_se3(se_path)

			x = torch.einsum('bjik,bj->bjik', x, se_path)

		x = self.conv3(x)
		x = self.bn3(x)
		#x = self.tanh3(x)

		if ((self.stride == 1) and (self.ch_in == self.ch_out)):
			#https://github.com/tensorflow/models/blob/1d057dfc32f515a63ab1e23fd72052ab2a954952/research/slim/nets/mobilenet/conv_blocks.py#L163
			# according to the link if strid is 1 and the input output dimntions are equal we have residual path
			#x = self.dropout_res(x) + residual
			x += residual
			x = self.bn_res(x)
		
		x = self.tanh_res(x)

		return x


class EfficientNet_my(nn.Module):
	def __init__(self, num_classes=1000, scale=0,  majority="BBBBBBB", backprop='normalConv', optimizer='Adam', dataset='imagenet'):
		super(EfficientNet_my, self).__init__()
		self.dataset = dataset
		if dataset == 'imagenet':
			strides = [1, 2, 2, 2, 1, 2, 1]
			d_ins = [112, 112, 56, 28, 14, 14, 7]

			# according to https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py
			# according to Archive paper
			#strides = [1, 2, 2, 1, 2, 2, 1]
			#d_ins = [112, 112, 56, 28, 28, 14, 7]
		elif dataset in ['cifar10', 'cifar100', 'svhn']:
			d_ins = [16, 16, 8, 4, 2, 2, 1]
			strides = [1, 2, 2, 2, 1, 2, 1]
		else:
			raise Exception('efficientnet_binary is not ready for anydataset rather than imagenet and cifar10/100')


		channels = [16, 24, 40, 80, 112, 192, 320]
		layers = [1, 2, 2, 3, 3, 4, 1]
		expansions = [1, 6, 6, 6, 6, 6, 6]        
		kernels = [3, 3, 5, 3, 5, 5, 3]


		self.conv1 = BinarizeConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32, eps=1e-03, momentum=0.01)
		self.tanh1 = nn.Hardtanh(inplace=True)

		self.layer0 = self._make_layer(d_ins[0],          32, channels[0], layers[0], kernel=kernels[0], expansion=expansions[0], stride=strides[0], majority=majority[0], backprop=backprop)
		self.layer1 = self._make_layer(d_ins[1], channels[0], channels[1], layers[1], kernel=kernels[1], expansion=expansions[1], stride=strides[1], majority=majority[1], backprop=backprop)
		self.layer2 = self._make_layer(d_ins[2], channels[1], channels[2], layers[2], kernel=kernels[2], expansion=expansions[2], stride=strides[2], majority=majority[2], backprop=backprop)
		self.layer3 = self._make_layer(d_ins[3], channels[2], channels[3], layers[3], kernel=kernels[3], expansion=expansions[3], stride=strides[3], majority=majority[3], backprop=backprop)
		self.layer4 = self._make_layer(d_ins[4], channels[3], channels[4], layers[4], kernel=kernels[4], expansion=expansions[4], stride=strides[4], majority=majority[4], backprop=backprop)
		self.layer5 = self._make_layer(d_ins[5], channels[4], channels[5], layers[5], kernel=kernels[5], expansion=expansions[5], stride=strides[5], majority=majority[5], backprop=backprop)
		self.layer6 = self._make_layer(d_ins[6], channels[5], channels[6], layers[6], kernel=kernels[6], expansion=expansions[6], stride=strides[6], majority=majority[6], backprop=backprop)
		
		self.conv2 = BinarizeConv2d(channels[6], 1280, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(1280, eps=1e-03, momentum=0.01)
		self.tanh2 = nn.Hardtanh(inplace=True)

		self.avgpool = nn.AvgPool2d(7)
		self.bn3 = nn.BatchNorm2d(1280, eps=1e-03, momentum=0.01)
		self.tanh3 = nn.Hardtanh(inplace=True)

		self.fc = BinarizeLinear(1280, num_classes)
		self.bn4 = nn.BatchNorm1d(num_classes, eps=1e-03, momentum=0.01)
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


	def _make_layer(self, d_in, ch_in, ch_out, repeat, kernel=3, expansion=1, stride=1, SE_en=True, majority="B", backprop='normalConvs'):
		
		layers = []    							
		layers.append(MobileNetBlock(d_in, ch_in, ch_out, kernel=kernel, expansion=expansion, stride=stride, SE_en=SE_en, majority=majority, backprop=backprop))

		for i in range(1, repeat):
			layers.append(MobileNetBlock(int(d_in/stride), ch_out, ch_out, kernel=kernel, expansion=expansion, stride=1, SE_en=SE_en, majority=majority, backprop=backprop))

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

		if self.dataset in ['imagenet']:
			x = self.avgpool(x)
			x = self.bn3(x)
			x = self.tanh3(x)
		
		x = x.view(x.size(0), -1)

		x = self.fc(x)
		x = self.bn4(x)
		x = self.logsoftmax(x)

		return x

		
def efficientnet_binary(**kwargs):
	num_classes, dataset, majority, backprop, efficientnet_scale, pretrained = map(kwargs.get, ['num_classes', 'dataset', 'majority', 'backprop', 'efficientnet_scale', 'pretrained'])

	supported_datasets = ['imagenet', 'cifar10', 'cifar100', 'svhn']
	if dataset in supported_datasets:
		return EfficientNet_my(num_classes=num_classes, scale=efficientnet_scale, majority=majority, backprop=backprop, dataset=dataset)

	else:
		raise Exception('efficientnet_binary is not ready for anydataset rather than imagenet')

## Quick Start

Requirements:
```
virtualenv -p /usr/bin/python3 venv3
pip install torch torchvision numpy bokeh tensorboardX==1.6
```

Build Majority Cuda:
```
cd models/majority_cuda/ && python setup.py install
```

For binarized VGG on cifar10, run:
```
python main_binary.py --model vgg_cifar10_binary --dataset cifar10 --majority BBBBB --padding 1 --gpus=1
```

For binarized **VGG+maj3** on cifar10, run:
```
python main_binary.py --model vgg_cifar10_binary --dataset cifar10 --majority MMMMM --padding 1 --gpus=1
```

For binarized **CNV+maj3** on cifar10, run:
```
python main_binary.py --model cnv_cifar10_binary --dataset cifar10 --majority MMBBB --padding 0 --gpus=1
```

For resume
```
--resume results/cnv_cifar10_binary_MMBBB_pad=0/model_best.pth.tar 
```

For MNIST
```
python main_mnist.py --gpus=1 --majority-enable --network=LFC --epochs=100
--majority-enable --> True/False flag
--network=LFC/SFC
```

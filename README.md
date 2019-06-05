## Quick Start

Requirements:
```
virtualenv -p /usr/bin/python3 venv3
pip install torch torchvision numpy bokeh tensorboardX==1.6
```


For binarized VGG on cifar10, run:
```
python main_binary.py --model vgg_cifar10_binary --save vgg_binary --dataset cifar10 --gpus=1 --epochs=200
```

For binarized **VGG+maj3** on cifar10, run:
```
python main_binary.py --model vgg_cifar10_maj3 --save vgg_maj3 --dataset cifar10 --batch-size=50 --backprop=normalConv --gpus=1 --epochs=200
```

For binarized **CNV+maj3** on cifar10, run:
```
python main_binary.py --model cnv_cifar10_maj3 --save cnv_maj3 --dataset cifar10 --batch-size=50 --backprop=normalConv --gpus=1 --epochs=200
```


For resume
```
 --resume results/cnv_maj3/model_best.pth.tar 
```


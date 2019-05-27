## Quick Start

For binarized VGG on cifar10, run:
```
python main_binary.py --model vgg_cifar10_binary --save vgg_binary --dataset cifar10
```

For binarized **VGG+maj3** on cifar10, run:
```
python main_binary.py --model vgg_cifar10_maj3 --save vgg_maj3 --dataset cifar10 --batch-size=50 --backprop=normalConv
```

For binarized **CNV+maj3** on cifar10, run:
```
python main_binary.py --model cnv_cifar10_maj3 --save cnv_maj3 --dataset cifar10 --batch-size=50 --backprop=normalConv --gpus=1
```

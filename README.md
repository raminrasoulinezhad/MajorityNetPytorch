
## The repository is ready to do:
* Cifar10/100+svhn --> cnv_binary(BM) / vgg_binary(BM) / resnet_binary(BM)
* MNIST --> SFC / LFC


## Requirements:
0- Install CUDA Driver() + CUDA (10.1) + CuDNN (10.1)

Please look at the appendix A

https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one
https://developer.nvidia.com/cuda-downloads

After installation
```
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib/
``` 


1- creat a new environment using:
```
virtualenv -p /usr/bin/python3 venv3
pip install torch torchvision numpy bokeh tensorboardX==1.6 scipy
```
2- Build Majority Cuda:
```
cd models/majority3_cuda/
python setup.py install
```
3- Don't forget to check the Dataset directory:
```
data.py for main.py: --> _DATASETS_MAIN_PATH = '/home/ramin/Download/Datasets'
main_mnist.py        --> '../data'
```

## Quick Start
For binarized VGG on cifar10/100, run:
```
python main_binary.py --model vgg_binary --dataset cifar10 --majority BBBBB --padding 1 --gpus=1
```

For binarized **VGG+maj3** on cifar10/100, run:
```
python main_binary.py --model vgg_binary --dataset cifar10 --majority MMMMM --padding 1 --gpus=1
```

For binarized **CNV+maj3** on cifar10/100, run:
```
python main_binary.py --model cnv_binary --dataset cifar10 --majority MMBBB --padding 0 --gpus=1
```

For binarized **CNV+maj3** on cifar10/100, run:
```
python main_binary.py --model resnet_binary --dataset cifar10 --majority BMM --gpus=0 

```

For resume
```
--resume results/cnv_binary_MMBBB_pad=0/model_best.pth.tar 
```

For MNIST
```
python main_mnist.py --gpus=1 --majority-enable --network=LFC --epochs=100
--majority-enable --> True/False flag
--network=LFC/SFC
```


# Appendix A

To remove previous verions (if you want):

	https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one

To install Driver:

	http://www.askaswiss.com/2019/01/how-to-install-cuda-9-cudnn-7-ubuntu-18-04.html


To download CUDA:
	
	https://developer.nvidia.com/cuda-downloads

To install CuDNN

	https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

After installation (you should export this library path as well):

	export LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib/









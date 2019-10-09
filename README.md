## The repository is ready to do:
* Cifar10/100+svhn --> cnv_binary(BM) / vgg_binary(BM) / resnet_binary(BM)
* MNIST --> SFC / LFC


## Requirements:
0- Install CUDA Driver + CUDA (10.1) + CuDNN (10.1): (Please look at the appendix A)

	https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one
	https://developer.nvidia.com/cuda-downloads

After installation

	export LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib/


1- creat a new environment using:
```
virtualenv -p /usr/bin/python3 venv3 && source venv3/bin/activate && pip install torch torchvision numpy bokeh tensorboardX==1.6 scipy pandas
```
2- Build Majority Cuda:
```
cd models/majority_cuda/ && python setup.py install && cd ./../../
```
3- Don't forget to check the Dataset directory:
```
'./../Datasets'
```

## Quick Start
For binarized EfficientNet on ImageNet:

	python main_binary.py --model=efficientnet_binary --dataset=imagenet --majority=BBBBBBB --gpus=0 -b=50 --pretrained --start-epoch=181 --epochs=500 --batch-size-val-ratio=5
	--majority=BBBBBBB is not working yet 
	-b=50 is not the original value (256) but it is the best that we could fit
	--pretrained to use the pretrained weights 
	--start-epoch=181 to just start from a good point
	--epochs=500 to extend the epoch limitations
	--batch-size-val-ratio=5 is used to help memory management --> higher batch size for training and less for validation (by factor of 5)

For binarized **VGG+maj3** on cifar10/cifar100/svhn, run:

	python main_binary.py --model=vgg_binary --dataset=cifar10 --majority=BMBBM+M --padding=1 --gpus=0


For binarized **CNV+maj3** on cifar10/cifar100/svhn, run:

	python main_binary.py --model=cnv_binary --dataset=cifar10 --majority MMBBM+B --padding=0 --gpus=1


For binarized **ResNet+maj3** on cifar10/cifar100/svhn, run:

	python main_binary.py --model=resnet_binary --dataset=cifar10 --majority=BMM --gpus=0 



For resume

	--resume results/cnv_binary_MMBBB_pad=0/model_best.pth.tar 


For MNIST

	python main_mnist.py --gpus=0 --network=LFC --epochs=100
	python main_mnist.py --gpus=0 --majority-enable --majority-apx --majority_size=9 --network=LFC --epochs=100
	
	flags:
	--majority-enable  		--> True/False (Default: False)
	--majority_size=3/5/7/9 --> (Default: 3)
	--majority-apx    		--> True/False (Default: False)
	--network=LFC/SFC		--> (Default: SFC)



# Appendix A

To deal with "Failed installation of package breaks apt-get" 

	sudo apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken

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

# Appendix B

For EfficientNet we used the following project (48th commit in 29th Augest 2019 (commit: de40cbfec8244a6ddbb367fd491d700ecc2eef85), Downloaded in 9th October 2019): 

	https://github.com/lukemelas/EfficientNet-PyTorch


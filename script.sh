# this file is prepared to check the effect of MajorityFC layers on SFC and LFC

python main_mnist.py --gpus=0 --network=SFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=3 --network=SFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=5 --network=SFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=7 --network=SFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=9 --network=SFC --epochs=100

python main_mnist.py --gpus=0 --network=LFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=3 --network=LFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=5 --network=LFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=7 --network=LFC --epochs=100
python main_mnist.py --gpus=0 --majority-enable --majority_size=9 --network=LFC --epochs=100

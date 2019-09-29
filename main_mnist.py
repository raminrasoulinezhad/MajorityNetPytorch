from __future__ import print_function
import os
import time
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
#from models.binarized_modules import  Binarize,Ternarize,Ternarize2,Ternarize3,Ternarize4,HingeLoss
from models.binarized_modules import  Binarize,HingeLoss,Maj3FC,MajFC

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=1,type=int,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--majority-enable', action='store_true', default=False,
                    help='enableing MajFC')
parser.add_argument('--majority_size', default=3,type=int,
                    help='majority_size used for training - e.g 3,5,7,9')
parser.add_argument('--preprocess', action='store_true', default=False,
                    help='default preprocess (False) or (-1,1) preprocess (True)')
parser.add_argument('--network', default='SFC',
                    help='default is SFC. it can be LFC also')
parser.add_argument('--dataset_dir', default='../Datasets/MNIST_main_mnist',
                    help='default is ../Datasets/MNIST_main_mnist ')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if (not args.preprocess):
    pre_mean, pre_u = 0.1307, 0.3081
else:
    pre_mean, pre_u = 0.5, 0.5

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.dataset_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((pre_mean,), (pre_u,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.dataset_dir, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((pre_mean,), (pre_u,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self, majority_enable=False, majority_size=3, network='SFC'):
        super(Net, self).__init__()

        self.majority_enable = majority_enable
        self.majority_size = majority_size
        
        self.infl_ratio=1

        if (network == 'SFC'):
            base = 256
            self.base = (base+(majority_size-(base % majority_size))) if ((base % majority_size != 0) & (majority_enable)) else base

        elif network == 'LFC':
            base = 1024
            self.base = (base+(majority_size-(base % majority_size))) if ((base % majority_size != 0) & (majority_enable)) else base
        else: 
            raise Exception('ramin: Choose the network: SFC or LFC by --network tag')

        input_size = 784
        self.input_size = (input_size+(majority_size-(input_size % majority_size))) if ((input_size % majority_size != 0) & (majority_enable)) else input_size
        self.majority_pad = ((majority_size-(input_size % majority_size))) if ((input_size % majority_size != 0) & (majority_enable)) else 0        

        print('Majority FCs: ' + str(majority_enable))
        print('majority_size: '+ str(majority_size))
        print('majority_pad: ' + str(self.majority_pad))
        print('base: ' + str(base) + ' ==> ' + str(self.base))
        print('input_size: '+ str(input_size) + ' ==> ' + str(self.input_size))

        if (majority_enable):
            self.fc1 = MajFC(self.input_size, self.base*self.infl_ratio, majority_size=self.majority_size)
        else:
            self.fc1 = BinarizeLinear(self.input_size, self.base*self.infl_ratio)
        
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(self.base*self.infl_ratio)
        
        if (majority_enable):
            self.fc2 = MajFC(self.base*self.infl_ratio, self.base*self.infl_ratio, majority_size=self.majority_size)
        else:
            self.fc2 = BinarizeLinear(self.base*self.infl_ratio, self.base*self.infl_ratio)

        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(self.base*self.infl_ratio)

        if (majority_enable):
            self.fc3 = MajFC(self.base*self.infl_ratio, self.base*self.infl_ratio, majority_size=self.majority_size)
        else:
            self.fc3 = BinarizeLinear(self.base*self.infl_ratio, self.base*self.infl_ratio)

        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(self.base*self.infl_ratio)
        self.fc4 = nn.Linear(self.base*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        
        if (self.majority_enable):
            x = F.pad(x, (self.majority_pad,0), "constant", 1.0)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)

assert ((args.majority_size % 2) == 1) , 'defined majority_size should be and odd number such as 3,5,7,9'
assert (args.majority_size > 1) , 'defined majority_size should be and odd number such as 3,5,7,9'

model = Net(majority_enable=args.majority_enable, majority_size=args.majority_size, network=args.network)
if args.cuda:
    torch.cuda.set_device(args.gpus)	
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (100. * float(correct) / float(len(test_loader.dataset)))



def main():

    if (args.majority_enable):
        maj_string = 'Maj' + str(args.majority_size)
    else:
        maj_string = 'Normal'

    save_name = args.network + "_" + maj_string + "_MNIST_" + 'epoch=' + str(args.epochs)
    save_path = os.path.join(args.results_dir, save_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        # append datatime
        tim = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        overwrite = input ("Directory {} already exists. Would you like to overwrite (y/n): ".format(save_path))
        if (overwrite == "y"):
            save_path = save_path
        else:
            save_path = save_path+"_{}".format(tim)
            os.makedirs(save_path)


    results_file = open(save_path + "/results.log","w+")

    acc_best = 0.0
    epoch_best = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)    
        acc_recent = test()
        if (acc_recent > acc_best):
            acc_best = acc_recent
            epoch_best = epoch
        print('\nBest answer: Epoch={}, Accuracy= ({:.2f}%)\n'.format(epoch_best, acc_best))
        results_file.write('Epoch={},Best answer: Epoch={}, Accuracy= ({:.2f}%)\n'.format(epoch, epoch_best, acc_best))


if __name__ == '__main__':
    main()

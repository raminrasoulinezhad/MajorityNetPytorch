import math
from torch import nn
from torch.autograd import Function
import torch
import time
import torch.optim as optim

######### Loading the new command #########
## Static loading - by running the following command in terminal: >> python setup.py install
import maj_cuda

## Dynamic loading and compiling (use the two following lines)
#from torch.utils.cpp_extension import load
#maj_cuda = load(name='maj', sources=['maj_cuda.cpp', 'maj_cuda_kernel.cu'])
#############################################

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class majFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, backprop, padding, kernel_size):
        input = input.transpose(1,2).transpose(2,3)

        if (backprop == 'normalConv'):
        	outputs = maj_cuda.forward_NBP(input.contiguous(), weight, padding)
        	output = outputs[0]
        	ctx.save_for_backward(input.contiguous(), weight, padding)

        else:
        	outputs = maj_cuda.forward(input.contiguous(), weight)
        	output, inter = outputs
        	ctx.save_for_backward(input.contiguous(), weight, inter, padding)

        ctx.backprop = backprop
        return output.transpose(2,3).transpose(1,2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.transpose(1,2).transpose(2,3)
        	
        if (ctx.backprop == 'normalConv'):
            outputs = maj_cuda.backward_NBP(grad_output.contiguous(), *ctx.saved_variables)
        else:
        	outputs = maj_cuda.backward(grad_output.contiguous(), *ctx.saved_variables)

        d_input, d_weight = outputs

        d_backprop, d_padding, d_kernel_size = None, None, None
        return d_input.transpose(2,3).transpose(1,2), d_weight, d_backprop, d_padding, d_kernel_size


class MajConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, bias=False, backprop='majority', padding=1):
        super(MajConv, self).__init__()
        
        self.backprop = backprop

        self.kernel_size = kernel_size
        self.padding = (torch.ones(1).cuda() * math.floor(self.kernel_size/2) * padding).to(torch.int)
        print ("kernel size: " + str(self.kernel_size))
        print ("padding: " + str(self.padding))

        assert not((padding == 0 ) and (not (backprop == 'normalConv'))), "Ramin: backprop=majority & padding=0 is not supported !!"

        self.c_in = c_in
        self.c_out = c_out

        self.weight = torch.nn.Parameter(torch.empty(c_out, kernel_size, kernel_size, c_in))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.c_in * self.kernel_size * self.kernel_size)
        for param in self.parameters():
            param.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        
        self.weight.data=Binarize(self.weight.org)

        return majFunction.apply(input, self.weight, self.backprop, self.padding, self.kernel_size)




############################################################
############################################################
############################################################
def mse_loss(input, target):
    return ((input - target) ** 2).sum() / input.data.nelement()

############################################################
assert torch.cuda.is_available()
torch.cuda.set_device(0)

layer = 4
list_Win = [32,16,16,8,8]
list_Cout = [128,256,256,512,512]
list_Cin = [128,128,256,256,512]

B = 64
kernel_size = 5

Win = list_Win[layer]
Hin = Win
Cin = list_Cin[layer]
Cout = list_Cout[layer]

X = torch.randint(0, 2 , (B, Cin, Win, Hin), dtype=torch.float).mul_(2.0).add_(-1).cuda()
############################################################
maj3 = MajConv(Cin, Cout, kernel_size, backprop='normalConv', padding=0).cuda()

optimizer = optim.SGD(maj3.parameters(), lr=0.01, momentum=0.9)
############################################################

forward, backward = 0, 0
iter_counter = 10
for iter in range(iter_counter):

    start = time.time()
    
    output = maj3(X)
    print(output.size())
    torch.cuda.synchronize()
    forward += time.time() - start
    
    start = time.time()
    loss = mse_loss(output, torch.zeros_like(output))
    print (loss)
    optimizer.zero_grad()
    (loss/100000).backward()
    torch.cuda.synchronize()
    optimizer.step()
    backward += time.time() - start
    
    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/(iter+1), backward * 1e6/(iter+1)))

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iter_counter, backward * 1e6/iter_counter))




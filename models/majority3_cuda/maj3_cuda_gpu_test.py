import math
from torch import nn
from torch.autograd import Function
import torch
import time
import torch.optim as optim

torch.manual_seed(46)

######### Loading the new command #########
# Static loading - by running: # python setup.py install
import maj3_cuda

# Dynamic loading and compiling 
#from torch.utils.cpp_extension import load
#maj3_cuda = load(name='maj3', sources=['maj3_cuda.cpp', 'maj3_cuda_kernel.cu'])
#############################################




class maj3Function(Function):
    @staticmethod
    def forward(ctx, input, weights):
    #def forward(ctx, input, weights, bias, old_h, old_cell):
        
        input = input.transpose(1,2).transpose(2,3)
        outputs = maj3_cuda.forward(input.contiguous(), weights)
        #outputs = maj3_cuda.forward(input, weights, bias, old_h, old_cell)

        #print(outputs[0])
        output, inter = outputs
        #new_h, new_cell = outputs[:2]

        #variables = weights + inter
        ##variables = outputs[1:] + [weights]

        ctx.save_for_backward(input.contiguous(), weights, inter)
        #ctx.save_for_backward(*variables)
        ##ctx.save_for_backward(*variables)

        output = output.transpose(2,3).transpose(1,2)

        return output
        #return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_output):
    #def backward(ctx, grad_h, grad_cell):
        grad_output = grad_output.transpose(1,2).transpose(2,3)
        #input, weights, inter = ctx.saved_tensors
        #print('d_out', grad_output)
        outputs = maj3_cuda.backward(grad_output.contiguous(), *ctx.saved_variables)
        
        #outputs = maj3_cuda.backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
		
        d_input, d_weights = outputs
        #d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        #print ('d_in', d_input, 'd_weights', d_weights)
        
        return d_input.transpose(2,3).transpose(1,2), d_weights
        #return d_input, d_weights, d_bias, d_old_h, d_old_cell

class maj3Function_NBP(Function):
    @staticmethod
    def forward(ctx, input, weights):
        input = input.transpose(1,2).transpose(2,3)
        output = maj3_cuda.forward_NBP(input.contiguous(), weights)
        ctx.save_for_backward(input.contiguous(), weights)
        imm = output[0].transpose(2,3).transpose(1,2)
        return imm

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.transpose(1,2).transpose(2,3)
        outputs = maj3_cuda.backward_NBP(grad_output.contiguous(), *ctx.saved_variables)
        d_input, d_weights = outputs
        return d_input.transpose(2,3).transpose(1,2), d_weights


class Maj3(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, bias=False, backprop='majority'):
        super(Maj3, self).__init__()
        
        self.backprop = backprop

        self.k_w = kernel_size
        self.k_h = kernel_size
        self.k_w_pad = math.floor(self.k_w/2)
        self.k_h_pad = math.floor(self.k_h/2)
        
        self.c_in = c_in
        self.c_out = c_out

        self.weights = torch.nn.Parameter(torch.empty(c_out, self.k_w , self.k_h , c_in))
        self.reset_parameters()

    def reset_parameters(self):
        #stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            #weight.data.uniform_(-stdv, +stdv)
            weight.data = torch.rand(weight.data.shape, dtype=torch.float).mul_(2.0).add_(-1) # device=cuda_device,
            #weight.data = torch.ones(weight.data.shape, device=cuda_device, dtype=torch.float)

    def forward(self, input):
        # it sems that quantization should be applied on both weights and inputs here
        
        if (self.backprop == 'normalConv'):
        	return maj3Function_NBP.apply(input, self.weights)
        else:
        	return maj3Function.apply(input, self.weights)


"""
############################################################
############################################################
############################################################
def mse_loss(input, target):
    return ((input - target) ** 2).sum() / input.data.nelement()

############################################################
assert torch.cuda.is_available()
cuda_device = torch.device('cuda:0')  # device object representing GPU

B = 50
Win = 32
Hin = 32
Cin = 128

Cout = 256
Kh = 3
Kw = 3
Khpad = math.floor(Kh/2)
Kwpad = math.floor(Kw/2)

X = torch.randint(0, 2 , (B, Win, Hin, Cin), device=cuda_device, dtype=torch.float).mul_(2.0).add_(-1)
#X = torch.ones((B, Win, Hin, Cin), device=cuda_device, dtype=torch.float) * -1 
############################################################
maj3 = Maj3(Cout, Kw, Kh, Cin, backprop='normalConv').to(device=cuda_device)

optimizer = optim.SGD(maj3.parameters(), lr=0.01, momentum=0.9)
############################################################

forward = 0
backward = 0
iter_counter = 5
for _ in range(iter_counter):
    start = time.time()
    
    #print(maj3.weights.sum())

    #print('input', X)
    #print('weights', maj3.weights)
    output = maj3(X)
    #print('output', output)

    #print(output.max())
    #print(output.min())
    torch.cuda.synchronize()
    forward += time.time() - start
    
    start = time.time()

    optimizer.zero_grad()
    
    loss = mse_loss(output, torch.zeros_like(output))
    print (loss)
    
    (loss/100000).backward()
    torch.cuda.synchronize()

    optimizer.step()
    #print(maj3.weights)

    
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iter_counter, backward * 1e6/iter_counter))
"""

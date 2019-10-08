import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor, quant_mode='det',  params=None, numBits=8):
    #tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        # I used clamp here to be sure Quantization is happending.
        tensor=tensor.mul(2**(numBits-1)).round().clamp(-2**(numBits-1),2**(numBits-1)).div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 784:
            input.data = Binarize(input.data)
        else:
            #input.data = Quantize(input.data, numBits=8)
            input.data = Binarize(input.data)

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        else:
            input.data = Quantize(input.data, numBits=8)

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class MultiQuantizedConv2d_h(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(MultiQuantizedConv2d_h, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        else:
            input.data = Quantize(input.data, numBits=8)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        num_channel = self.weight.data.size(1)
        num_channel_h = int ((num_channel/8)*7)

        self.weight.data[:,0:num_channel_h-1,:,:] = Binarize(self.weight.org[:,0:num_channel_h-1,:,:])
        self.weight.data[:,num_channel_h:num_channel-1,:,:] = Quantize(self.weight.org[:,num_channel_h:num_channel-1,:,:], numBits=8)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


# Pure Python based FC using Majority-3
class Maj3FC(nn.Module):
    def __init__(self, c_in, c_out, bias=False):
        super(Maj3FC, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.weight = torch.nn.Parameter(torch.empty(c_out, c_in))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.c_out * self.c_in)
        for param in self.parameters():
            param.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        if input.size(1) != 784:
            input.data = Binarize(input.data)
        else:
            input.data = Binarize(input.data)
            #input.data = Quantize(input.data, numBits=8)
        
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        
        self.weight.data=Binarize(self.weight.org)

        c_in_d3 = int(self.c_in/3)
        inter_maj3 = torch.sum(torch.einsum('bj,cj->bcj', input, self.weight).reshape([-1, c_in_d3, 3]), 2).reshape([-1, c_in_d3])
        out = torch.sum(torch.clamp(inter_maj3, min=-1.0, max=1.0), 1).mul_(2.25).reshape([-1, self.c_out])

        #if not self.bias is None:
        #    self.bias.org=self.bias.data.clone()
        #    out += self.bias.view(1, -1).expand_as(out)

        return out

# Pure Python based FC using Majority-n
class MajFC(nn.Module):
    def __init__(self, c_in, c_out, majority_size=3, majority_apx=False, bias=False):
        super(MajFC, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.majority_size = majority_size
        self.majority_apx = majority_apx

        self.weight = torch.nn.Parameter(torch.empty(c_out, c_in))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.c_out * self.c_in)
        for param in self.parameters():
            param.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        if input.size(1) != 784:
            input.data = Binarize(input.data)
        else:
            input.data = Binarize(input.data)
            #input.data = Quantize(input.data, numBits=8)
        
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        
        self.weight.data=Binarize(self.weight.org)

        if (self.majority_apx):
            if (self.majority_size == 3):
                raise Exception('majority-3 has no approximation computing')
            elif (self.majority_size == 5):
                assert (self.c_in % 5 == 0), "C_in is not dividable by 5 in MajFC-5apx layer"
                c_in_d5 = int(self.c_in/5)
                inter_chunked = torch.chunk(torch.einsum('bj,cj->bcj', input, self.weight).reshape([-1, c_in_d5, 5]), 2, dim=2)
                inter_maj3 = torch.clamp(torch.sum(inter_chunked[0].reshape([-1, c_in_d5, 3]), 2).reshape([-1, c_in_d5, 1]), min=-1.0, max=1.0)
                # 3.75 = 2.25 * (5/3)
                out = torch.sum(torch.clamp(torch.sum(torch.cat((inter_maj3, inter_chunked[1]), dim=2),2), min=-1.0, max=1.0).reshape([-1, c_in_d5]), 1).mul_(3.75).reshape([-1, self.c_out])
                
            elif (self.majority_size == 7):
                assert (self.c_in % 7 == 0), "C_in is not dividable by 7 in MajFC-5apx layer"
                c_in_d7 = int(self.c_in/7)
                inter_chunked = torch.chunk(torch.einsum('bj,cj->bcj', input, self.weight).reshape([-1, c_in_d7, 7]), 3, dim=2)

                inter_maj3_1 = torch.clamp(torch.sum(inter_chunked[0].reshape([-1, c_in_d7, 3]), 2).reshape([-1, c_in_d7, 1]), min=-1.0, max=1.0)
                inter_maj3_2 = torch.clamp(torch.sum(inter_chunked[1].reshape([-1, c_in_d7, 3]), 2).reshape([-1, c_in_d7, 1]), min=-1.0, max=1.0)
                # 5.25 = 2.25 * (7/3)
                out = torch.sum(torch.clamp(torch.sum(torch.cat((inter_maj3_1, inter_maj3_2, inter_chunked[2]), dim=2), 2), min=-1.0, max=1.0).reshape([-1, c_in_d7]), 1).mul_(5.25).reshape([-1, self.c_out])

            elif (self.majority_size == 9):
                assert (self.c_in % 9 == 0), "C_in is not dividable by 9 in MajFC-9apx layer"
                c_in_d3 = int(self.c_in/3)
                c_in_d9 = int(self.c_in/9)
                inter_maj3 = torch.sum(torch.einsum('bj,cj->bcj', input, self.weight).reshape([-1, c_in_d3, 3]), 2).reshape([-1, c_in_d9, 3])
                inter_maj9 = torch.sum(torch.clamp(inter_maj3, min=-1.0, max=1.0), 2).reshape([-1, c_in_d9])
                out = torch.sum(torch.clamp(inter_maj9, min=-1.0, max=1.0), 1).mul_(2.25*3).reshape([-1, self.c_out])
            else:
                raise Exception('approximate majority-{} is not supported'.format(self.majority_size))    
        else:
            c_in_d3 = int(self.c_in/self.majority_size)
            inter_maj3 = torch.sum(torch.einsum('bj,cj->bcj', input, self.weight).reshape([-1, c_in_d3, self.majority_size]), 2).reshape([-1, c_in_d3])
            out = torch.sum(torch.clamp(inter_maj3, min=-1.0, max=1.0), 1).mul_(2.25).reshape([-1, self.c_out])

        #if not self.bias is None:
        #    self.bias.org=self.bias.data.clone()
        #    out += self.bias.view(1, -1).expand_as(out)

        return out

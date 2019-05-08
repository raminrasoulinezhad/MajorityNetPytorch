## remaining tasks:
    ## 1- initial values
    ## 2- quantization for forward


import torch
import torch.nn.functional as F
import math
import time
import torch.optim as optim

class Maj3(torch.nn.Module):
    def __init__(self, c_out, k_w, k_h, c_in):
        super(Maj3, self).__init__()

        self.k_w = k_w
        self.k_h = k_h
        self.k_w_pad = math.floor(k_w/2)
        self.k_h_pad = math.floor(k_h/2)
        
        self.c_in = c_in
        self.c_out = c_out

        self.weights = torch.nn.Parameter(torch.empty(c_out, k_w * k_h * c_in))
    
        self.reset_parameters()

    def reset_parameters(self):
        #stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            #weight.data.uniform_(-stdv, +stdv)
            weight.data = torch.randint(0, 2 , weight.data.shape, device=cuda_device, 
                dtype=torch.float).mul_(2.0).add_(-1)

    def forward(self, input):
        # it sems that quantization should be applied on both weights and inputs here

        input_shape = input.shape
        w_in = input_shape[1]
        h_in = input_shape[2]

        input_paded = F.pad(input, (0,0,self.k_h_pad,self.k_h_pad,self.k_w_pad,self.k_w_pad), 
            mode='constant', value=-1)

        input_unfolded = input_paded.unfold(1,self.k_w,1).unfold(2,self.k_h,1).reshape(
            [-1,self.k_w*self.k_h*self.c_in])
        
        inter_maj3 = torch.sum(torch.einsum('bj,aj->baj', input_unfolded, self.weights).reshape(
            [-1,self.k_w,self.k_h*self.c_in]), 1)
        output = torch.sum(torch.clamp(inter_maj3, min=-1.0, max=1.0).mul_(3.0), 1).reshape(
            [-1, w_in, h_in, self.c_out])
        
        return output


def mse_loss(input, target):
    return ((input - target) ** 2).sum() / input.data.nelement()
##############################
assert torch.cuda.is_available()
cuda_device = torch.device('cuda:0')  # device object representing GPU

B = 50
Win = 32
Hin = 32
Cin = 4

Cout = 8
Kh = 3
Kw = 3
Khpad = math.floor(Kh/2)
Kwpad = math.floor(Kw/2)

X = torch.randint(0, 2 , (B, Win, Hin, Cin), 
    device=cuda_device, dtype=torch.float).mul_(2.0).add_(-1)



maj3 = Maj3(Cout, Kw, Kh, Cin).to(device=cuda_device)

optimizer = optim.SGD(maj3.parameters(), lr=0.01, momentum=0.9)


forward = 0
backward = 0
iter_counter = 10
for _ in range(iter_counter):
    start = time.time()
    output = maj3(X)
    #print(output[0,0,0,:])
    
    torch.cuda.synchronize()
    forward += time.time() - start

    start = time.time()

    optimizer.zero_grad()
    loss = mse_loss(output, torch.zeros_like(output))
    print (loss)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(
    forward * 1e6/iter_counter, backward * 1e6/iter_counter))

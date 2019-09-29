from torch.utils.cpp_extension import load

# to dynamically load the maj-3 functions from c++ and cuda files
maj3_cuda = load(
    'maj3_cuda', ['maj3_cuda.cpp', 'maj3_cuda_kernel.cu'], verbose=True)
help(maj3_cuda)

# to dynamically load the maj-n functions from c++ and cuda files
maj_cuda = load(
    'maj_cuda', ['maj_cuda.cpp', 'maj_cuda_kernel.cu'], verbose=True)
help(maj_cuda)

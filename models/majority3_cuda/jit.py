from torch.utils.cpp_extension import load
#lltm_cuda = load(
#    'lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
#help(lltm_cuda)
#

maj3_cuda = load(
    'maj3_cuda', ['maj3_cuda.cpp', 'maj3_cuda_kernel.cu'], verbose=True)
help(maj3_cuda)

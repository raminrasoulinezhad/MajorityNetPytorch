from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#############################
# maj3_cuda setup configurations
#############################

setup(
    name='maj3_cuda',
    ext_modules=[
        CUDAExtension('maj3_cuda', [
            'maj3_cuda.cpp',
            'maj3_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


#############################
# maj_cuda setup configurations
#############################

setup(
    name='maj_cuda',
    ext_modules=[
        CUDAExtension('maj_cuda', [
            'maj_cuda.cpp',
            'maj_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#setup(
#    name='lltm_cuda',
#    ext_modules=[
#        CUDAExtension('lltm_cuda', [
#            'lltm_cuda.cpp',
#            'lltm_cuda_kernel.cu',
#        ]),
#    ],
#    cmdclass={
#        'build_ext': BuildExtension
#    })
#

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


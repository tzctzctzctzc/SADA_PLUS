from setuptools import setup, find_packages
import torch
import os
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CUDAExtension

def get_extensions():
    ext_modules = [
        CUDAExtension(
            name='models.TadTR.ops.roi_align.Align1D',
            sources=[
                'models/TadTR/ops/roi_align/roi_align_cuda.cpp',
                'models/TadTR/ops/roi_align/roi_align_kernel.cu'
            ],
            extra_compile_args = [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ])
    ]
    return ext_modules

setup(
    name='TempFusion',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    version='0.0.1',
    license='MIT',
    description='',
    author='David Pujol-Perich and Germán Barquero',
    author_email='david.pujolperich@gmail.com',
    url='',
    keywords=[
        'artificial intelligence',
        'computer vision',
        'temporal action localization',
        'diffusion'
    ],
    install_requires=[
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

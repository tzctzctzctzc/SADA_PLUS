"""用明确指定的CUDA工具链编译作者原始算子；编译时不初始化GPU。"""
import os
from pathlib import Path
import sys
import site
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension,CppExtension,BuildExtension

code=Path(__file__).resolve().parents[1]
src=code/'vendor/digit/models/digit/ops/src'
cuda_headers=[str(p) for root in site.getsitepackages() for p in Path(root).glob('nvidia/*/include')]
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
setup(name='digit_benchmark_extensions',ext_modules=[
    CUDAExtension('MultiScaleTemporalDeformableAttention',
        [str(p) for p in sorted(src.glob('*.cpp'))]+[str(p) for p in sorted((src/'cpu').glob('*.cpp'))]+[str(p) for p in sorted((src/'cuda').glob('*.cu'))],
        include_dirs=[str(src)]+cuda_headers,define_macros=[('WITH_CUDA',None)],
        extra_compile_args={'cxx':[], 'nvcc':['-DCUDA_HAS_FP16=1','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__']}),
    CppExtension('nms_1d_cpu',[str(code/'vendor/dyfadet/libs/utils/csrc/nms_cpu.cpp')],extra_compile_args=['-fopenmp'])
],cmdclass={'build_ext':BuildExtension})

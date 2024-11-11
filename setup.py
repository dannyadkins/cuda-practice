from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

# Find all CUDA source files
cuda_sources = []
for cuda_file in Path('algos').rglob('*.cu'):
    cuda_sources.append(str(cuda_file))

for cuda_file in Path('algos').rglob('*_binding.cpp'): 
    cuda_sources.append(str(cuda_file))

setup(
    name='cuda-practice',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='cuda_kernels',
            sources=cuda_sources,
            extra_compile_args={'cxx': ['-O2'],
                              'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
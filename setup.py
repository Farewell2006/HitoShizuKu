# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:20:08 2026

@author: Togawa Sakiko
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(    
          name='HitoShizuKu',
          ext_modules=[CUDAExtension(name='HitoShizuKu', 
                                     sources=['kernels/GaussianRatserization.cu',
                                              'kernels/cppwarpper/gaussian_rasterization_warpper.cpp',
                                              'kernels/3dgs_forward.cu','kernels/3dgs_structure.cu','kernels/3dgs_backward.cu'], 
                                     include_dirs=[os.path.abspath("kernels/")],
                                     extra_compile_args={
                                         "cxx":["-O2","-std=c++14"],
                                         "nvcc":["-O2","-arch=sm_89","-std=c++17","-Xcompiler", "/Zc:__cplusplus"],
                                         }
                                     )
                       ],cmdclass={"build_ext":BuildExtension}
      
      )

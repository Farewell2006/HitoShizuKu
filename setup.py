# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:52:13 2026

@author: TogawaSakiko
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
import os


setup(
      name='HitoShizuKu',
      ext_modules=[CUDAExtension(name='HitoShizuKu', sources=['kernel/warpper/warpper.cpp','kernel/rasterizer.cu','kernel/rasterizerfwd.cu',
                                                        'kernel/render/renderfwd.cu','kernel/render/render.cu','kernel/render/sah.cu',
                                                        'kernel/render/renderbwd.cu',
                                                        ],
                                 include_dirs=[os.path.abspath("kernel/"),os.path.abspath("kernel/render/")],
                                 extra_compile_args={"cxx":["-O2","-std=c++14"],"nvcc":["-O2","-arch=sm_89","-std=c++17"]})
                   ],
                  cmdclass={'build_ext':BuildExtension}
      
      )



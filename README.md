# HitoShizuKu
A deep learning toolkit used for study purpose.  
## Functions
### Differentiable Gaussian Rasterizer
#### References
This work draws on the architectural design of the open-source project [https://github.com/graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) as a reference for its basic code structure. Our implementation is constructed entirely from scratch based on a thorough comprehension of the original 3D Gaussian Splatting paper and the logical framework of the reference project. No source code from the original repository has been directly reused in this codebase.
#### Features

1. Interactivte

<div align="center">
  <img src="imgs/gsrasterizer.gif" width="500" alt="Gaussian_rasterizer">
</div>

2. Differentiable
   
   Tested with naive torch implementation

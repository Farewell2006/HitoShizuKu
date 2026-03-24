# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:26:46 2026

@author: 25488
"""

import os
import sys

root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from geometry.dmt import marching_tetrahedra
import torch


if __name__=="__main__":
    device='cpu'
    
    vertices=torch.Tensor([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]],device=device).float()
    tet=torch.Tensor([[0,1,3,5],[4,5,0,6],[0,3,2,6],[5,3,6,7],[0,5,3,6]],device=device).long()
    
    sdf=torch.Tensor([[1,1,1,1,1,1,1,1],[1,1,1,1,-1,1,1,1],[1,1,1,1,-1,-1,1,1],[1,1,1,1,-0.5,-0.7,1,1]]).float()
    
    v,f,idx=marching_tetrahedra(vertices, tet, sdf[3])
    print(v)
    print(f)
    print(idx)




































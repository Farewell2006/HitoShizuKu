# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:16:54 2026

@author: Togawa Sakiko
"""

import torch
from .transformation import from_lookat
import numpy as np


class CameraExtrinsic:
    def __init__(self,azim:float,elev:float,up,dist:float):
        self.azim=azim
        self.elev=elev
        self.up=up
        self.dist=dist
        
        self.R,self.T=from_lookat(self.azim,self.elev, self.up, self.dist)
        
class CameraIntrinsic:
    def __init__(self, foccal_length:tuple, principal_point:tuple,image_size:tuple):
        self.focal_length=foccal_length
        self.principal_point=principal_point
        self.image_size=image_size
        
class PerspectiveCamera:
    def __init__(self,Extrinsic:CameraExtrinsic,Intrinsic:CameraIntrinsic):
        self.Extrinsic=Extrinsic
        self.Intrinsic=Intrinsic
        
        self.fx=self.Intrinsic.focal_length[0]
        self.fy=self.Intrinsic.focal_length[1]
        
        self.cx=self.Intrinsic.principal_point[0]
        self.cy=self.Intrinsic.principal_point[1]
        
        self.K=np.array([[self.fx,0.0,self.cx,0.0],
                         [0.0,self.fy,self.cy,0.0],
                         [0.0,0.0,0.0,1.0],
                         [0.0,0.0,1.0,0.0]])
        
        self.view_matrix=self.get_view_matrix()
        
        self.tan_fovx=0.5*self.Intrinsic.image_size[0]/self.fx
        self.tan_fovy=0.5*self.Intrinsic.image_size[1]/self.fy
        
        
    def get_view_matrix(self,):
        
        V=np.eye(4)
        V[:3,:3]=self.Extrinsic.R
        V[:3,3]=self.Extrinsic.T
        return V
        
    def get_full_proj(self,):
        return self.K@ self.view_matrix
    def cam_to_world(self,):
        V=np.eye(4)
        V[:3,:3]=self.Extrinsic.R.T
        V[:3,3]=- V[:3,:3]@self.Extrinsic.T
        return V
    def cam_pos(self,):
        
        return -self.Extrinsic.R.T@self.Extrinsic.T
    
    
if __name__=='__main__':
    
    dim=256
    img_size = (dim, dim)

    num_views = 32
    azims = np.linspace(-180, 180, num_views)
    elevs = np.linspace(-180, 180, num_views)
    up=(0,-1,0)
    dist=8.0
    
    
    
    
    
    
    
    
    
    
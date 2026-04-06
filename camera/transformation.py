# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:32:41 2026

@author: Togawa Sakiko
"""

import torch
import numpy as np
import math

def Euler_Matrix(thetaX,thetaY,thetaZ):
    ''' 给定关于x,y,z轴的旋转角度， 输出对应的 3 x 3 旋转矩阵'''
    cosx=math.cos(thetaX)
    cosy=math.cos(thetaY)
    cosz=math.cos(thetaZ)
    
    sinx=math.sin(thetaX)
    siny=math.sin(thetaY)
    sinz=math.sin(thetaZ)
    
    M=[[cosy*cosz,-cosy*sinz,siny],
       [cosz*sinx*siny+cosx*sinz,cosx*cosz-sinx*siny*sinz,-cosy*sinx],
       [-cosx*cosz*siny+sinx*sinz,cosz*sinx+cosx*siny*sinz,cosx*cosy]]
    
    M=torch.Tensor(M)
    return M

def Angle_Axis_Rotation(theta:float, axis):
    '''给定旋转轴和旋转角，输出对应的旋转矩阵'''
    ux=axis[0]
    uy=axis[1]
    uz=axis[2]
    
    cos=math.cos(theta)
    sin=math.sin(theta)
    
    M=[[cos+ux*ux*(1-cos),ux*uy*(1-cos)-uz*sin,ux*uz*(1-cos)+uy*sin],
       [uy*ux*(1-cos)+uz*sin,cos+uy*uy*(1-cos),uy*uz*(1-cos)-ux*sin],
       [uz*ux*(1-cos)-uy*sin,uz*uy*(1-cos)+ux*sin,cos+uz*uz*(1-cos)]]
    
    M=torch.Tensor(M)
    return M

def from_lookat(azim,elev,up,dist):
    '''给定旋转角azim和俯仰角elev 以及上方向量up和与原点的距离, 求解旋转矩阵R和平移T'''
    azim=math.radians(azim)
    elev=math.radians(elev)
    
    cosa=math.cos(azim)
    sina=math.sin(azim)
    cosb=math.cos(elev)
    sinb=math.sin(elev)
    
    f=-np.array([sina*cosb,sinb,cosa*cosb])
    
    cam_pos=dist*(-f)
    right=np.cross(up, f)
    right=right/np.linalg.norm(right)
    
    up=np.cross(f, right)
    
    R=np.zeros((3,3))
    R[:,0]=right
    R[:,1]=up
    R[:,2]=f
    
    T=-cam_pos@ R
    return R.T,T
    
    
    

if __name__=="__main__":
    
    azim=-180
    elev=-180
    up=(0,-1,0)
    dist=6.0
    R,T=from_lookat(azim, elev, up,dist)
    print(R,T)
    
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:16:54 2026

@author: Togawa Sakiko
"""

import torch
from torch.nn import Parameter
import math
import glm


class CameraExtrinsic:
    '''记录相机的外蕴性质，比如位置，局部坐标的轴向。'''
    def __init__(self,position=torch.ones((3,)),direction=torch.eye(n=3), device='cpu',):
        '''
            初始化一个默认位置和姿态的相机    
        
            position: [3]
            direction: [3,3] 相机局部坐标的三个方向，按照x轴,y轴,z轴排列
            
            它们可能是潜在的可优化的参数，所以是torch.nn.Parameter
        '''
        self.position=Parameter(position).to(device)
        self.direction=Parameter(direction).to(device)
        self.view_matrix=torch.eye(4).to(device)
        
    def from_lookat(self, eye:torch.Tensor, up:torch.Tensor, at:torch.Tensor,device='cpu'):
        '''
            从lookat信息生成外蕴参数
            
            eye: [3] 世界坐标系下 相机的坐标
            up: [3]  世界坐标系下 相机的上方对应的方向向量(按右手系，只要给出这个向量和 eye到at的向量，就能得出右手系的第三个向量)
            at: [3] 世界坐标系下 相机对准的点
            
        '''
        #注意相机看向的方向和局部坐标系的z轴是反向的(计算机图形学中大部分规定相机看向负z).
        forward=(eye-at)/torch.linalg.norm(eye-at)
        #right是up和forward的叉积, 注意up在左.
        right=torch.cross(up, forward,dim=-1)
        #R是世界坐标系到相机坐标系的旋转.
        R=torch.stack([right,up,forward],dim=0)
        #然后把世界坐标系的原点转移到相机坐标系.
        u=R.T@(at-eye)
        
        view_matrix=torch.eye(4,device=device)
        view_matrix[:3,:3]=R
        view_matrix[:3,3]=u
        self.view_matrix[:3,:3]=R
        self.view_matrix[:3,3]=R@(at-eye)
    
    def from_view_matrix(self,m):
        '''
            从视图矩阵生成外蕴参数
            
        '''
        
    def from_camera_pose(position,direction):
        '''从位姿生成外蕴参数，和初始化格式一致'''
        
    def ViewMatrix(self,):
        return self.view_matrix
        
        
class CameraIntrinsic:
    '''记录相机的内蕴性质，比如焦距，视角，屏幕的宽和高'''
    def __init__(self,focal_length,principal_point,image_shape,):
        ''' near和far都是正数'''
        
        self.l=left
        self.r=right
        self.t=top
        self.b=bottom
        self.n=near
        self.f=far
        
        self.clip_matrix=torch.Tensor([[2*self.n/(self.r-self.l),0,(self.l+self.r)/(self.r-self.l),0],
                                       [0,2*self.n/(self.t-self.b),(self.b+self.t)/(self.t-self.b),0],
                                       [0,0,-1*(self.f+self.n)/(self.f-self.n),-2*self.f*self.n/(self.f-self.n)],
                                       [0,0,-1,0]])
    
class Camera:
    '''通过外蕴和内蕴性质初始化一个相机对象'''
    def __init__(self,extrinsic:CameraExtrinsic,intrinsic:CameraIntrinsic):
        self.extrinsic=extrinsic
        self.intrinsic=intrinsic
        
    def world_to_view(self,points:torch.Tensor):
        
        #变成齐次坐标
        points=torch.concat([points,torch.ones(size=(points.shape[0],1),device=points.device)],dim=-1)
        
        view_matrix=self.extrinsic.ViewMatrix()
        
        #(1,4,4) x (N,4,1) -> (N,4,1) 保留齐次坐标格式
        view_coordinate=torch.matmul(view_matrix.unsqueeze(0), points.unsqueeze(-1))
        return view_coordinate
    
    def view_to_clip(self,view):
        return torch.matmul(self.intrinsic.clip_matrix.unsqueeze(0),view)
    
    def world_to_clip(self,points):
        points=torch.concat([points,torch.ones(size=(points.shape[0],1),device=points.device)],dim=-1).unsqueeze(-1)
        
        view_matrix=self.extrinsic.ViewMatrix()
        
        #(1,4,4) x (N,4,1) -> (N,4,1) 保留齐次坐标格式
        view_coordinate=torch.matmul(view_matrix.unsqueeze(0), points)
        #print(self.intrinsic.clip_matrix)
        
        return torch.matmul(self.intrinsic.clip_matrix.unsqueeze(0),view_coordinate)
    
    def world_to_ndc(self,points):
        p=self.world_to_clip(points)
        
        p=p[:,:3]/p[:,-1]
        return p
        

if __name__=="__main__":
    eye=torch.Tensor([3,3,3]).float()
    at=torch.zeros_like(eye)
    up=torch.Tensor([-2,1,1])/math.sqrt(6)
    
    Ex=CameraExtrinsic()
    Ex.from_lookat(eye, up, at)
    In=CameraIntrinsic(left=-5.0, right=5.0, top=5.0, bottom=-5.0, near=1, far=7.0)
    camera=Camera(extrinsic=Ex, intrinsic=In)
    #print(camera.intrinsic.clip_matrix )
    
    
    
    
    vertices=torch.Tensor([[0,0,5],[0,5,0],[5,0,0]])
    
    ndc_v=camera.world_to_ndc(vertices)
    print(ndc_v)
    
    
    eye=glm.vec3(2,2,2)
    up=glm.vec3(-2,1,1)
    at=glm.vec3(0,0,0)
    
    m=glm.lookAt(eye, at, up)
    
    
    
    
    
    
    
    
    
    
    
    
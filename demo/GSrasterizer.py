# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:48:04 2026

@author: TogawaSakiko
"""

import os
os.add_dll_directory(r"D:\anaconda\Lib\site-packages\torch\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")
import HitoShizuKu
import torch
import numpy as np
import math
import sys
from pathlib import Path
import gc

import cv2
sys.path.append(str(Path(__file__).parent.parent))
from camera.camera import CameraExtrinsic,CameraIntrinsic,PerspectiveCamera
from render.Gaussian_rasterizer import load_pretained



class InteractionState:
    def __init__(self, init_azim=-180,init_elev=-180.0,init_dist=6.0):
        self.azim=init_azim
        self.elev=init_elev
        self.dist=init_dist
        self.last_mouse_pose=None
        self.last_button_pressed=None

def mouse_callback(event,x,y,flag,param):
    state=param
    if event==cv2.EVENT_LBUTTONDOWN:
        state.last_button_pressed=True
        state.last_mouse_pose=(x,y)
    elif event==cv2.EVENT_LBUTTONUP:
        state.last_button_pressed=False
    elif event==cv2.EVENT_MOUSEMOVE and state.last_button_pressed:
        dx=x-state.last_mouse_pose[0]
        dy=y-state.last_mouse_pose[1]
        state.azim-=0.3*dx
        state.elev-=0.3*dy
        
        #state.elev=np.clip(state.elev,-89.0,89.0)
        state.last_mouse_pose=(x,y)
    elif event==cv2.EVENT_MOUSEWHEEL:
        state.dist+=flag/12000.0*0.002
        state.dist=np.clip(state.dist, 0.1, 100)
        
if __name__=="__main__":
        
    state=InteractionState()
    
    device='cuda'
    dim=1024
    dist=6.0
    azim=-180
    elev=-180
    up=(0,-1,1)
   
    Ex=CameraExtrinsic(azim, elev, up, dist)
    In=CameraIntrinsic((5*dim/2,5*dim/2),(dim/2,dim/2),(dim,dim))
   
    cam=PerspectiveCamera(Ex,In)

    
    tan_fovx=cam.tan_fovx
    tan_fovy=cam.tan_fovy
    view_matrix=torch.Tensor(cam.view_matrix).to(device)
    proj_matrix=torch.Tensor(cam.get_full_proj()).to(device)
    
    
    device='cuda'
    dic=load_pretained()
    means3D=torch.Tensor(dic['xyz'])[:-1,:].to(device)
    
    
    rotation=torch.Tensor(dic['rotation']).to(device)
    rotation=torch.nn.functional.normalize(rotation,dim=-1)
    
    scale=torch.exp(torch.Tensor(dic['scale'])).to(device)
    
    sh=torch.Tensor(dic["sh"]).to(device)
    opacity=torch.nn.Sigmoid()(torch.Tensor(dic["opacity"])).to(device)
    
    
    color=torch.Tensor(dic['dc_colours']).to(device)
    background=torch.ones(size=[3,]).to(device)
    
    color1=torch.zeros(size=(3,dim,dim)).to(device)
    rendered=0
    
    
    
    
    cv2.namedWindow("Interactive Gaussian Rasterizer",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Interactive Gaussian Rasterizer", mouse_callback,param=state)
    
    fps=0
    prev_time=cv2.getTickCount()
    
    while True:
        
        Ex=CameraExtrinsic(state.azim, state.elev, up, state.dist)
        #In=CameraIntrinsic((5*dim/2,5*dim/2),(dim/2,dim/2),(dim,dim))
       
        cam=PerspectiveCamera(Ex,In)

        
        tan_fovx=cam.tan_fovx
        tan_fovy=cam.tan_fovy
        view_matrix=torch.Tensor(cam.view_matrix).to(device)
        proj_matrix=torch.Tensor(cam.get_full_proj()).to(device)
        
        
        rendered, color1,_,_,_=HitoShizuKu.GaussianRasterizationForward(background,means3D,color,opacity,scale,rotation,1.0,
                                                                        view_matrix,proj_matrix,tan_fovx,tan_fovy,dim,dim)
        
        
        
        #del GEO,BIN,IMG
        #gc.collect()
        #torch.cuda.empty_cache()
        print(f"已分配显存: {torch.cuda.memory_allocated()/1024**2:.2f}MB, 缓存显存: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        
        #break
        res=color1.detach().permute(1,2,0).cpu().numpy()
        res=cv2.cvtColor(res,cv2.COLOR_RGB2BGR)
        current_time=cv2.getTickCount()
        fps=cv2.getTickFrequency()/(current_time-prev_time)
        prev_time=current_time
        cv2.putText(res, f"FPS: {fps:.1f}", (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.imshow("Interactive Gaussian Rasterizer", res)
        
        key=cv2.waitKey(1)& 0xFF
        # ESC退出 r键重置
        if key==27:
            break
        elif key==ord('r'):
            state.azim=-180
            state.elev=-180
            state.dist=6.0
    cv2.destroyAllWindows()
            
    
    
    
    
    
    
    
    
    
    
    
    

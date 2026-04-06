# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:19:38 2026

@author: Togawa Sakiko
"""

import os
os.add_dll_directory(r"D:\anaconda\Lib\site-packages\torch\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")
import HitoShizuKu
import torch
from plyfile import PlyData
import numpy as np
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from camera.camera import PerspectiveCamera,CameraExtrinsic,CameraIntrinsic

import matplotlib.pyplot as plt


def load_pretained():
    SH_C0 = 0.28209479177387814
    max_sh_degree=3
    
    plydata=PlyData.read(str(Path(__file__).parent)+'./test_data/chair.ply')
    xyz=np.stack([np.array(plydata.elements[0]['x']),np.array(plydata.elements[0]['y']),np.array(plydata.elements[0]['z'])],axis=1)
    
    opacity=np.array(plydata.elements[0]['opacity'])
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots.astype(np.float32)
    scales = scales.astype(np.float32)
    opacity = opacity.astype(np.float32)
    shs = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(len(features_dc), -1)
    ], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)

    dc_vals = shs[:, :3]
    dc_colours = np.maximum(dc_vals * SH_C0 + 0.5, np.zeros_like(dc_vals))

    output = {
        "xyz": xyz, "rotation": rots, "scale": scales,
        "sh": shs, "opacity": opacity, "dc_colours": dc_colours
    }
    
    return output

def visualize(res):
    res=res.permute(1,2,0).detach().cpu().numpy()
    plt.figure(figsize=(8, 8))

    plt.imshow(res)
    plt.title("rendered image", fontsize=12)
    plt.axis("off")
    plt.show()

if __name__=="__main__":
    
    device='cuda'
    dim=256
    dist=6.0
    azim=-180
    elev=-180
    up=(0,-1,1)
   
    Ex=CameraExtrinsic(azim, elev, up, dist)
    In=CameraIntrinsic((5*dim/2,5*dim/2),(dim/2,dim/2),(dim,dim))
   
    cam=PerspectiveCamera(Ex,In)

    
    cam_pos=[0.0,0.0,6.0]
    tan_fovx=cam.tan_fovx
    tan_fovy=cam.tan_fovy
    
    cam_pos=torch.tensor(cam_pos).to(device)
    view_matrix=torch.Tensor(cam.view_matrix).to(device)
    proj_matrix=torch.Tensor(cam.get_full_proj()).to(device)
    
    
    device='cuda'
    dic=load_pretained()
    means3D=torch.Tensor(dic['xyz'])[:30,:].to(device)
    
    #把均值放到原点附近
    #means3D=means3D-means3D.mean(dim=0,keepdim=True)
    
    rotation=torch.Tensor(dic['rotation']).to(device)
    rotation=torch.nn.functional.normalize(rotation,dim=-1)
    
    scale=torch.exp(torch.Tensor(dic['scale'])).to(device)
    
    sh=torch.Tensor(dic["sh"]).to(device)
    opacity=torch.nn.Sigmoid()(torch.Tensor(dic["opacity"])).to(device)
    
    
    color=torch.Tensor(dic['dc_colours']).to(device)
    
    
    
    background=torch.ones(size=[3,]).to(device)
    color1=torch.zeros(size=(3,dim,dim)).to(device)
    rendered=0
    rendered, color1,geometryBuffer,BinningBuffer,ImageBuffer=HitoShizuKu.GaussianRasterizationForward(background,means3D,
                                                                                                       color,opacity,scale,rotation,1.0,
                                                                                                       view_matrix,proj_matrix,tan_fovx,tan_fovy,
                                                                                                       dim,dim)
    
    #visualize(color1)
    
    
    dO=torch.ones_like(color1)
    
    dL_dmeans3D,dL_dcolor,dL_dscale,dL_dopacity,dL_dq=HitoShizuKu.GaussianRasterizationBackward(background,means3D,
                                                                                              color,opacity,scale,rotation,1.0,
                                                                                               view_matrix,proj_matrix,tan_fovx,tan_fovy,
                                                                                               dim,dim,dO,
                                                                                                geometryBuffer,BinningBuffer,ImageBuffer,rendered)
    
    
    #print(dL_dopacity)
    #print(dL_dcolor)
    #print(dL_dscale)
    #print(dL_dq)
    print(dL_dmeans3D)
    
    
    
    
    
    
    
    
    
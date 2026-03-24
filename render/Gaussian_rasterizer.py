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
def load_pretained():
    SH_C0 = 0.28209479177387814
    max_sh_degree=3
    plydata=PlyData.read('./test_data/chair.ply')
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

def get_projection_transform( 
    focal_length=1.0,
    principal_point=((0.0, 0.0),),
    image_size=None,
    in_ndc=False,
    znear=0.01,
    zfar=100.0
 ):
    """
    构造PyTorch3D风格的透视投影矩阵
    参数:
        focal_length: 焦距，标量或长度为2的序列（f_x, f_y）
        principal_point: 主点，格式为((c_x, c_y),)
        image_size: 图像尺寸，格式为(H, W)
        in_ndc: 是否使用NDC坐标系
        znear: 近裁剪面距离
        zfar: 远裁剪面距离
    返回:
        proj_matrix: 4×4投影矩阵，右乘约定
    """
    H, W = image_size
    c_x, c_y = principal_point[0]
    
    # 处理焦距为标量的情况（x/y方向焦距相等）
    if isinstance(focal_length, (int, float)):
        f_x = f_y = focal_length
    else:
        f_x, f_y = focal_length
    
    # 初始化4x4矩阵
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    
    if in_ndc:
        # NDC坐标系公式
        proj_matrix[0, 0] = f_x / (W / 2)
        proj_matrix[1, 1] = f_y / (H / 2)
        proj_matrix[0, 2] = (W/2 - c_x) / (W/2)
        proj_matrix[1, 2] = (H/2 - c_y) / (H/2)
    else:
        # 像素坐标系公式
        proj_matrix[0, 0] = 2 * f_x / W
        proj_matrix[1, 1] = 2 * f_y / H
        proj_matrix[0, 2] = 1 - 2 * c_x / W
        proj_matrix[1, 2] = 1 - 2 * c_y / H
    
    # 深度相关项（两种场景通用）
    proj_matrix[2, 2] = -(zfar + znear) / (zfar - znear)
    proj_matrix[2, 3] = -1.0
    proj_matrix[3, 2] = -2 * zfar * znear / (zfar - znear)
    
    return proj_matrix.T

def compute_alpha(opacity,power):
    exp=torch.where(power>0.0,0.0,torch.exp(power))
    
    
    return opacity.unsqueeze(-1)*exp

def Visualization(colors,cov2d,means2d,depth,opacity,dim):
    
    non_neg_ind=depth>=0
    non_neg_ind=torch.nonzero(non_neg_ind).squeeze(-1)
    
    de_val=depth[non_neg_ind]
    _,ind=torch.sort(de_val)
    ind = non_neg_ind[ind]
    
    
    colors=colors[ind]
    cov2d=cov2d[ind]
    means2d=means2d[ind]
    depth=depth[ind]
    opacity=opacity[ind]
    
    
    
    xs, ys = torch.meshgrid(torch.arange(dim), torch.arange(dim), indexing="xy")
    points_2D = torch.stack((xs.flatten(), ys.flatten()), dim = 1)  # (H*W, 2)
    points_2D = points_2D.to(cov2d.device)
    
    N=means2d.shape[0]
    means2d=means2d
   
    means_2D=means2d.unsqueeze(1)
    cov_2D_inverse=torch.zeros(size=(N,2,2),device=means2d.device)
    
    
    cov_2D_inverse[:,0,0]=cov2d[:,0]
    cov_2D_inverse[:,1,1]=cov2d[:,2]
    cov_2D_inverse[:,1,0]=cov2d[:,1]
    cov_2D_inverse[:,0,1]=cov2d[:,1]
    power = torch.bmm(-0.5*(points_2D-means_2D),cov_2D_inverse)
    power = torch.sum(power*(points_2D-means_2D),dim=-1)# (N, H*W)
    
    alpha=compute_alpha(opacity, power).reshape((N,dim,dim))
    
    
    alpha=torch.minimum(alpha, torch.full_like(alpha, 0.99))
    alpha=torch.where(alpha<1.0/255.0,0.0,alpha)
    
    T=torch.ones(size=(1,dim,dim),device=alpha.device)
    T=torch.cat([T,1.0-alpha],dim=0)
    T=torch.cumprod(T[:-1,], dim=0)
    T=torch.where(T<1e-4,0.0,T)
    
    #变成[N,1,H,W]
    alpha=alpha.unsqueeze(1)
    T=T.unsqueeze(1)
    
    
    #变成[N,3,1,1]
    colors=colors.unsqueeze(-1).unsqueeze(-1)
    
    
    res=torch.sum(alpha*T*colors,dim=0)
    res=res.reshape((3,dim,dim)).cpu().numpy()
    
    import matplotlib.pyplot as plt
 
    # 假设你的numpy数组名为img_np，形状是 [3, H, W]
    img_np = res  # 示例数组
     
    # 第一步：把通道从第一维转到最后一维，变成 [H, W, 3]
    img_vis = np.transpose(img_np, (1, 2, 0))  
     
    # 第二步：可视化
    plt.imshow(img_vis)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()
    
def compute_view_matrix(dist,azim,elev,up):
    azim_rad = math.radians(azim)
    elev_rad = math.radians(elev)
    
    # 计算相机位置
    x = dist * math.cos(elev_rad) * math.sin(azim_rad)
    y = dist * math.sin(elev_rad)
    z = dist * math.cos(elev_rad) * math.cos(azim_rad)
    cam_pos = torch.tensor([x, y, z], dtype=torch.float32)
    
    # 计算相机基向量
    forward = torch.nn.functional.normalize(-cam_pos, dim=0)  # 指向原点
    right = torch.nn.functional.normalize(torch.cross(torch.tensor(up, dtype=torch.float32), forward), dim=0)
    camera_up = torch.cross(forward, right)
    
    # 构造旋转矩阵
    R = torch.stack([right, camera_up, forward], dim=0)
    
    # 构造平移向量
    T = -cam_pos @ R
    
    # 组合为4×4矩阵
    view_matrix = torch.eye(4, dtype=torch.float32)
    view_matrix[:3, :3] = R
    view_matrix[3, :3] = T
    
    return view_matrix.T

if __name__=="__main__":
    
    device='cpu'
    dim=256
    dist=6.0
    azim=-180
    elev=-180
    view_matrix=compute_view_matrix(dist, azim, elev, up=(0,-1,0))
    
    proj1=get_projection_transform(focal_length=5.0*dim/2.0,principal_point=[[dim/2,dim/2]],image_size=(dim,dim))
    
    print(proj1@view_matrix)
    
    proj_matrix=[[ 6.4000e+02,  1.1190e-05,  0.0000e+00,  8.7423e-08],
         [-1.1190e-05, -6.4000e+02,  0.0000e+00, -8.7423e-08],
         [-1.2800e+02, -1.2800e+02,  0.0000e+00, -1.0000e+00],
         [ 7.6800e+02,  7.6800e+02,  1.0000e+00,  6.0000e+00]]
    
    
    cam_pos=[0.0,0.0,6.0]
    tan_fovx=0.2
    tan_fovy=0.2
    view_matrix=torch.tensor(view_matrix).to(device)
    proj_matrix=torch.tensor(proj_matrix).to(device).T
    cam_pos=torch.tensor(cam_pos).to(device)
    
    cam_pos=1
    
    device='cuda'
    dic=load_pretained()
    means3D=torch.Tensor(dic['xyz']).to(device)
    
    #把均值放到原点附近
    #means3D=means3D-means3D.mean(dim=0,keepdim=True)
    
    rotation=torch.Tensor(dic['rotation']).to(device)
    rotation=torch.nn.functional.normalize(rotation,dim=-1)
    
    scale=torch.exp(torch.Tensor(dic['scale'])).to(device)
    
    sh=torch.Tensor(dic["sh"]).to(device)
    opacity=torch.nn.Sigmoid()(torch.Tensor(dic["opacity"])).to(device)
    
    
    color=torch.Tensor(dic['dc_colours']).to(device)
    
    
    
    
    
    
    
    background=torch.zeros(size=[3,]).to(device)
    color1=torch.zeros(size=(3,dim,dim))
    #rendered, color1,position2d,cov2d=HitoShizuKu.GaussianRasterizationForward(background,means3D,color,opacity,scale,rotation,1.0,view_matrix,proj_matrix,tan_fovx,tan_fovy,dim,dim,sh,1,cam_pos)
    
   
    
    
    import matplotlib.pyplot as plt
 
    # 假设你的numpy数组名为img_np，形状是 [3, H, W]
    img_np = color1.detach().cpu().numpy()  # 示例数组
     
    # 第一步：把通道从第一维转到最后一维，变成 [H, W, 3]
    img_vis = np.transpose(img_np, (1, 2, 0))  
     
    # 第二步：可视化
    plt.imshow(img_vis)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
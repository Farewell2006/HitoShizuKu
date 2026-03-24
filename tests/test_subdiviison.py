# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:13:43 2026

@author: Togawa Sakiko
"""

import os
import sys

root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from geometry.subdivision import subidivision_trianglemesh_iters
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

def test_loop_subdivison_cube(iters):
    device = torch.device('cpu')  # 可改为cuda如果需要
 
    # 正方体8个顶点坐标 (x,y,z)，范围从0到1
    vertices = torch.Tensor([
        [0, 0, 0],  # 顶点0: 原点
        [1, 0, 0],  # 顶点1: x轴端点
        [1, 1, 0],  # 顶点2: xy平面对角
        [0, 1, 0],  # 顶点3: y轴端点
        [0, 0, 1],  # 顶点4: z轴端点
        [1, 0, 1],  # 顶点5: xz平面对角
        [1, 1, 1],  # 顶点6: 最远对角点
        [0, 1, 1]   # 顶点7: yz平面对角
    ], device=device).float()
     
    # 正方体6个面，每个面拆分为2个三角形，共12个三角面
    # 顶点索引按逆时针排列（正面朝向外部）
    faces = torch.Tensor([
        # 底面 (z=0)
        [0, 1, 2],
        [0, 2, 3],
        # 顶面 (z=1)
        [4, 5, 6],
        [4, 6, 7],
        # 前面 (y=0)
        [0, 1, 5],
        [0, 5, 4],
        # 后面 (y=1)
        [3, 2, 6],
        [3, 6, 7],
        # 左面 (x=0)
        [0, 3, 7],
        [0, 7, 4],
        # 右面 (x=1)
        [1, 2, 6],
        [1, 6, 5]
    ], device=device).long()
     
    alpha=torch.zeros(size=(vertices.shape[0],),device=device)+0.09
    
    v,f=subidivision_trianglemesh_iters(vertices, faces,alpha,iters)
    return v,f

def test_loop_subdivison_tet(iters):
    device = torch.device('cpu')  # 可根据需要改为cuda
    vertices = torch.Tensor([
        [0, 0, 0],   # 顶点0: 原点
        [1, 0, 0],   # 顶点1: x轴端点
        [0, 1, 0],   # 顶点2: y轴端点
        [0, 0, 1]    # 顶点3: z轴端点
    ], device=device).float()
    faces = torch.Tensor([
        [0, 1, 2],  # 底面（z=0平面）
        [0, 1, 3],  # 侧面1
        [0, 2, 3],  # 侧面2
        [1, 2, 3]   # 侧面3
    ], device=device).long()
    
    alpha=torch.zeros(size=(vertices.shape[0],),device=device)
    
    v,f=subidivision_trianglemesh_iters(vertices, faces,alpha,iters)
    return v,f
    
def test_loop_subdivison_ball(iters):
    device='cpu'
    vertices = torch.Tensor([
        [-0.52573111,  0.85065081,  0.        ],
        [ 0.52573111,  0.85065081,  0.        ],
        [-0.52573111, -0.85065081,  0.        ],
        [ 0.52573111, -0.85065081,  0.        ],
        [ 0.        , -0.52573111,  0.85065081],
        [ 0.        ,  0.52573111,  0.85065081],
        [ 0.        , -0.52573111, -0.85065081],
        [ 0.        ,  0.52573111, -0.85065081],
        [ 0.85065081,  0.        , -0.52573111],
        [ 0.85065081,  0.        ,  0.52573111],
        [-0.85065081,  0.        , -0.52573111],
        [-0.85065081,  0.        ,  0.52573111]
    ], device=device).float()
     
    # 正二十面体的20个三角面，所有面接近等边三角形
    faces = torch.Tensor([
        [0, 11,  5], [0,  5,  1], [0,  1,  7], [0,  7, 10], [0, 10, 11],
        [1,  5,  9], [5, 11,  4], [11, 10,  2], [10,  7,  6], [7,  1,  8],
        [3,  9,  4], [3,  4,  2], [3,  2,  6], [3,  6,  8], [3,  8,  9],
        [4,  9,  5], [2,  4, 11], [6,  2, 10], [8,  6,  7], [9,  8,  1]
    ], device=device).long()
    
    alpha=torch.zeros(size=(vertices.shape[0],),device=device)
    
    v,f=subidivision_trianglemesh_iters(vertices, faces,alpha,iters)
    return v,f
    
def visualize_mesh(vertices, faces, title="Mesh Visualization"):
    # 转换为numpy数组
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    x = verts_np[:, 0]
    y = verts_np[:, 1]
    z = verts_np[:, 2]
    
    # 创建3D绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三角形曲面
    ax.plot_trisurf(
        x, y, z, 
        triangles=faces_np,
        edgecolor='black',   # 网格边线颜色
        facecolor='lightblue', # 面填充颜色
        alpha=0.5,          # 半透明便于观察内部结构
        linewidth=0.5       # 边线宽度
    )
    
    # 绘制顶点标记
    ax.scatter(x, y, z, c='red', s=30, zorder=5)
    
    # 设置坐标轴
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    # 设置视角，方便观察正方体结构
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()






if __name__=="__main__":
    v,f=test_loop_subdivison_cube(iters=2)
    
    visualize_mesh(v, f)
    














































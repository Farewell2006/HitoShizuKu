# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:21:25 2026

@author: Togawa Sakiko
"""

import os
import sys

root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from geometry.dmt import volume_subdivision
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

def convert_tet_to_mesh(tet:torch.LongTensor):
    '''把tet变成mesh'''
    v1=tet[:,0]
    v2=tet[:,1]
    v3=tet[:,2]
    v4=tet[:,3]
    f1=torch.stack([v1,v2,v3],dim=-1)
    f2=torch.stack([v1,v2,v4],dim=-1)
    f3=torch.stack([v2,v3,v4],dim=-1)
    f4=torch.stack([v1,v3,v4],dim=-1)
    
    f=torch.concat([f1,f2,f3,f4],dim=0)
    return f

def visualization(v,f):
    
    f=convert_tet_to_mesh(f)
    
    v=v.detach().cpu().numpy()
    f=f.detach().cpu().numpy()
    
    x=v[:,0]
    y=v[:,1]
    z=v[:,2]
    
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111,projection='3d')
    
    ax.plot_trisurf(x,y,z,triangles=f,edgecolor='blue',facecolor='lightblue',alpha=0.1,linewidth=0.5)
    
    ax.scatter(x,y,z,c='red',s=30,zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    for idx,(x,y,z) in enumerate(v):
        ax.text(x+0.03,y+0.03,z+0.06,str(idx),color='darkred',fontsize=10,zorder=10)
    
    ax.view_init(elev=30,azim=45)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    device='cpu'
    vertices=torch.Tensor([[0.8,1,0.5],[0,1,0.5],[1,0,0.5],[0,0,1],[0.8,1,0]],device=device).float()
    tet=torch.Tensor([[0,1,2,3],[0,1,2,4]],device=device).long()
    
    sdf=torch.Tensor([[-1],[-1],[0.5],[0.5]],device=device)
    
    v,f,sdf=volume_subdivision(vertices, tet,None)
    #v,f,sdf=volume_subdivision(v,f,sdf)
    print(sdf)
    
    visualization(v,f)
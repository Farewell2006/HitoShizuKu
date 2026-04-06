# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:01:27 2026

@author: Togawa Sakiko
"""

'''     用torch模拟 Gaussian rasterizer的cuda实现 '''

import torch
import numpy as np
from Gaussian_rasterizer import load_pretained
import matplotlib.pyplot as plt

def getRect(means2D,radius):
    
    
    lx=torch.clamp_max(torch.floor((means2D[:,0]-radius)/16),16).to(torch.int32)
    ly=torch.clamp_max(torch.floor((means2D[:,1]-radius)/16),16).to(torch.int32)
    
    rx=torch.clamp_max(torch.floor((means2D[:,0]+radius+15)/16),16).to(torch.int32)
    ry=torch.clamp_max(torch.floor((means2D[:,1]+radius+15)/16),16).to(torch.int32)
    
    
    
    return lx,ly,rx,ry

class Gaussian:
    def __init__(self, means3D:torch.Tensor, scales:torch.Tensor, color:torch.Tensor, quant:torch.Tensor,
                 opacity:torch.Tensor, view_matrix:torch.Tensor, proj_matrix:torch.Tensor,background,
                 tan_fovx,tan_fovy,H,W):
        self.means3D=means3D.requires_grad_(True)
        self.scales=scales.requires_grad_(True)
        self.quant=quant.requires_grad_(True)
        self.color=color.requires_grad_(True)
        self.opacity=opacity.requires_grad_(True)
        
        
        self.background=background
        self.tan_fovx=tan_fovx
        self.tan_fovy=tan_fovy
        self.H=H
        self.W=W
        self.view_matrix=view_matrix
        self.proj_matrix=proj_matrix
        
        
        
    def compute_depth(self,points):
        '''  用view_matrix计算点的深度'''
        N=points.shape[0]
        #变成齐次坐标
        points=torch.cat([points, torch.ones(size=(N,1),device=points.device)],dim=-1)
        res=(self.view_matrix @ points.T).T
        
        depth=res[:,2]
        return depth
    
    def compute_screen_coor(self,points):
        N=points.shape[0]
        points=torch.cat([points, torch.ones(size=(N,1),device=points.device)],dim=-1)
        res=(self.proj_matrix @ points.T).T
        
        coor=res[:,:2]/res[:,-1].unsqueeze(-1)
        minus=torch.Tensor([[self.W,self.H]],device=points.device)
        return minus-coor
    
    def compute_cov3D(self,quaternions, scale):
        
            #先把四元数变成旋转矩阵
            w, x, y, z = quaternions.unbind(dim=-1)
        
            x2, y2, z2 = x * 2, y * 2, z * 2
            xx, xy, xz = x * x2, x * y2, x * z2
            yy, yz, zz = y * y2, y * z2, z * z2
            wx, wy, wz = w * x2, w * y2, w * z2
            
            R = torch.stack([
                1 - (yy + zz), xy - wz, xz + wy,
                xy + wz, 1 - (xx + zz), yz - wx,
                xz - wy, yz + wx, 1 - (xx + yy)
            ], dim=-1).reshape(-1, 3, 3)
            
            S=torch.diag_embed(scale,dim1=-2,dim2=-1)
            M=torch.bmm(R,S)
            
            cov3D=torch.bmm(M,M.transpose(-1, -2))
            
            return cov3D
    
    def compute_J(self,points):
        
            N=points.shape[0]
            points=torch.cat([points, torch.ones(size=(N,1),device=points.device)],dim=-1)
            view_point=(self.view_matrix @ points.T).T
            
            tx=view_point[:,0]
            ty=view_point[:,1]
            tz=view_point[:,2]
            
            
            tan_fovx=self.tan_fovx
            tan_fovy=self.tan_fovy
            
            limx=1.3*tan_fovx
            limy=1.3*tan_fovy
            
            tx=torch.clamp(tx/tz,-limx,limx)*tz
            ty=torch.clamp(ty/tz,-limy,limy)*tz
            
            J=torch.zeros(size=(N,2,3),device=points.device)
            
            fx=self.W/(2*tan_fovx)
            fy=self.H/(2*tan_fovy)
            
            J[:,0,0]=fx/tz
            J[:,1,1]=fy/tz
            J[:,0,2]=-fx*tx/tz/tz
            J[:,1,2]=-fy*ty/tz/tz
            
            return J
    
    def compute_cov2D(self,points):
        
            
            J=self.compute_J(points)
            V=self.view_matrix[:3,:3].unsqueeze(0).broadcast_to((points.shape[0],3,3))
            
            sigma3D=self.compute_cov3D(self.quant, self.scales)
            
            temp=torch.bmm(J,V)
            
            cov2D=torch.bmm(temp,sigma3D)
            cov2D=torch.bmm(cov2D,temp.transpose(-1, -2))
            
            cov2D[:,0,0]+=0.3
            cov2D[:,1,1]+=0.3
        
            return cov2D
    
    def DuplicateWithKeys(self,tiles_intersected,depth,num_rendered,lx,ly,rx,ry):
        
       
        N=tiles_intersected.shape[0]
        
        KeysUnsorted=np.zeros((num_rendered,2))
        PointIdx=np.zeros((num_rendered,1),dtype=np.int64)
        start=0
        
        lx=lx.cpu().tolist()
        ly=ly.cpu().tolist()
        rx=rx.cpu().tolist()
        ry=ry.cpu().tolist()
        
        for i in range(0,N):
            for j in range(lx[i],rx[i]):
                for k in range(ly[i],ry[i]):
                    block_idx=k*16+j
                    key=block_idx
                    KeysUnsorted[start][0]=key
                    KeysUnsorted[start][1]=depth[i]
                    PointIdx[start]=i
                    start+=1
        
        KeysUnsorted=torch.Tensor(KeysUnsorted,device=tiles_intersected.device)
        PointIdx=torch.Tensor(PointIdx,device=tiles_intersected.device)
        
        
        return KeysUnsorted, PointIdx
                    
                    
    def compute_grid(self,cov2D,screen2d):
        b=0.5*(cov2D[:,0,0]+cov2D[:,1,1])
        det=cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
        delta=torch.sqrt(torch.clamp_min(b*b-det,0.1))
        lambda1=b-delta
        lambda2=b+delta
        
        max_radius=(3.0*torch.sqrt(torch.max(lambda1,lambda2))).ceil().to(torch.int32)
        
        
        
        lx,ly,rx,ry=getRect(screen2d, max_radius)
        N=lx.shape[0]
        
        tiles_intersected=(rx-lx)*(ry-ly)
        tiles_intersected=torch.cumsum(tiles_intersected , dim=-1)
        
        rendered=tiles_intersected[-1].item()
        
        return tiles_intersected,rendered,lx,ly,rx,ry
        
        
        
        
    
    def preprocess(self,):
        
        screen2d=self.compute_screen_coor(self.means3D)
        
        
        cov2D=self.compute_cov2D(self.means3D)
        
        tiles_intersected,num_rendered,lx,ly,rx,ry=self.compute_grid(cov2D, screen2d)
        
        depth=self.compute_depth(self.means3D)
        
        KeysUnsorted,PointIdx=self.DuplicateWithKeys(tiles_intersected, depth, num_rendered, lx, ly, rx, ry)
        
        
        idx1=torch.argsort(KeysUnsorted[:,0],stable=True)
        temp=KeysUnsorted[idx1]
        idx2=torch.argsort(temp[:,1],stable=True)
        
        idx=idx1[idx2]
        
        SortedKeys=KeysUnsorted[idx][:,0].to(torch.int64)
        SortedPoints=PointIdx[idx].to(torch.int64)
        
        
        return SortedKeys,SortedPoints,cov2D,screen2d
        
    def getBlockRanges(self,SortedKeys,SortedPoints):
        N=SortedKeys.shape[0]
        BlockRange=dict()
        for i in range(0,N):
            key=SortedKeys[i].item()
            if key in BlockRange:
                BlockRange[key].append(SortedPoints[i].item())
            else:
                BlockRange[key]=[SortedPoints[i].item()]
                
        return BlockRange
    
    def compute2DInv(self,cov_2D):
        determinants = cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
        determinants = determinants[:, None, None]  # (N, 1, 1)

        cov_2D_inverse = torch.zeros_like(cov_2D)  # (N, 2, 2)
        cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
        cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
        cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
        cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]

        cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse
        
        
        return cov_2D_inverse
        
    def Splat(self,key,idx,cov2D,means2D):
        x=key%16
        y=key//16
        pix_minx=x*16
        pix_miny=16*y
        
        cov2D=cov2D[idx]
        means2D=means2D[idx]
        opacity=self.opacity[idx]
        color=self.color[idx]
        
        xs,ys=torch.meshgrid(torch.arange(pix_minx,pix_minx+16),torch.arange(pix_miny,pix_miny+16),indexing='xy')
        pix=torch.stack([xs.flatten(),ys.flatten()],dim=-1).unsqueeze(0)
        
        means2D=means2D.unsqueeze(1)
        
        #cov2D_inv=self.compute2DInv(cov2D)
        cov2D_inv=self.compute2DInv(cov2D)
        d=pix-means2D
        
        
        
        exponent=-0.5*d[:,:,0]*d[:,:,0]*cov2D_inv[:,0,0].unsqueeze(1)-0.5*d[:,:,1]*d[:,:,1]*cov2D_inv[:,1,1].unsqueeze(1)-0.5*d[:,:,0]*d[:,:,1]*cov2D_inv[:,0,1].unsqueeze(1)-0.5*d[:,:,0]*d[:,:,1]*cov2D_inv[:,1,0].unsqueeze(1)
        exponent_mask=exponent<=0
        exponent=torch.where(exponent>0,0.0,exponent)
        
        alpha=torch.exp(exponent)*opacity.unsqueeze(-1)
        alpha=torch.where(alpha>0.99,0.99,alpha)
        alpha=torch.where(alpha<1.0/255.0,0.0,alpha)
        
        T=torch.concat([torch.ones(size=(1,256),device=alpha.device),(1.0-alpha)],dim=0)
        
        #T=1.0-alpha
        T=torch.cumprod(T, dim=0)
        final_T=T[-1:]
        T=T[:-1,]
        #T=torch.where(torch.logical_or(T<0.0001, alpha<1.0/255), 0.0, T)
        
        
        T=torch.where(T<0.0001, 0.0, T)
        
        out=torch.zeros(size=(3,256),device=T.device)
        
        alpha=alpha.unsqueeze(-1)
        T=T.unsqueeze(-1)
        T_mask=T>0.0
        a_mask=alpha>1.0/255.0
        a_mask2=alpha<0.99
        
        a_mask=a_mask*a_mask2
        color=color.unsqueeze(1)
        
        out=(color*T*alpha).sum(dim=0).transpose(-1,-2)+(self.background.unsqueeze(0)*final_T.unsqueeze(-1)).transpose(-1,-2)
        out=out.reshape(3,16,16)
        #print(alpha[:,15*16+13])
        
        alpha=a_mask*alpha
        T=T*T_mask
        
        #然后计算微分 测试备用
        
        color=color*T_mask
        Aj=(color*alpha*T).sum(dim=0,keepdim=True)-(color*alpha*T).cumsum(dim=0)
        Aj=Aj/(1.0-alpha)
        dLda=color*T-Aj-(self.background.unsqueeze(0).unsqueeze(0)*final_T.unsqueeze(-1))/(1.0-alpha)
        
        g=(exponent_mask*torch.exp(exponent)).unsqueeze(-1)
        dLdo=(a_mask*g*dLda).sum(dim=[-1,-2])
        
        
        return out
            
    
    def workFlow(self,):
        SortedKeys,SortedPoints,cov2D,screen2d=self.preprocess()
        BlockRange=self.getBlockRanges(SortedKeys,SortedPoints)
        
        
        res=torch.ones((3,256,256))
        for key in BlockRange.keys():
            grid_x=key%16
            grid_y=key//16
            
            
            res[:,grid_y*16:grid_y*16+16,grid_x*16:grid_x*16+16]=self.Splat(key,BlockRange[key], cov2D, screen2d)
        
        
        return res
        
        
        

if __name__=="__main__":
    device='cpu'
    dim=256
    dist=6.0
    azim=-180
    elev=-180
    view_matrix=[[ 1.0000000e+00, -1.4997596e-32 ,-1.2246467e-16,  0.0000000e+00],
                 [ 0.0000000e+00, -1.0000000e+00 , 1.2246467e-16, -1.4695761e-15],
                 [-1.2246467e-16, -1.2246467e-16, -1.0000000e+00,  6.0000000e+00],
                 [ 0.0000000e+00,  0.0000000e+00 , 0.0000000e+00 , 1.0000000e+00]]
    
    proj_matrix=[[ 6.4000e+02,  1.1190e-05,  0.0000e+00,  8.7423e-08],
         [-1.1190e-05, -6.4000e+02,  0.0000e+00, -8.7423e-08],
         [-1.2800e+02, -1.2800e+02,  0.0000e+00, -1.0000e+00],
         [ 7.6800e+02,  7.6800e+02,  1.0000e+00,  6.0000e+00]]
    
    
    cam_pos=[0.0,0.0,6.0]
    tan_fovx=0.2
    tan_fovy=0.2
    view_matrix=torch.tensor(view_matrix).to(device)
    proj_matrix=torch.tensor(proj_matrix).to(device).T
    
    
    dic=load_pretained()
    means3D=torch.Tensor(dic['xyz'])[:30,:].to(device)
    
    #把均值放到原点附近
    #means3D=means3D-means3D.mean(dim=0,keepdim=True)
    
    rotation=torch.Tensor(dic['rotation'])[:30,:].to(device)
    rotation=torch.nn.functional.normalize(rotation,dim=-1)
    
    scale=torch.exp(torch.Tensor(dic['scale']))[:30,:].to(device)
    
    opacity=torch.nn.Sigmoid()(torch.Tensor(dic["opacity"]))[:30].to(device)
    
    color=torch.Tensor(dic['dc_colours'])[:30].to(device)
    
    background=torch.ones(size=[3,]).to(device)
    
    g=Gaussian(means3D, scale, color, rotation, opacity, view_matrix, proj_matrix,background, tan_fovx, tan_fovy, dim, dim)
    
    res=g.workFlow()
    
    
    loss=res.sum()
    loss.backward()
    
    #print(g.color.grad)
    #print(g.opacity.grad)
    print(g.means3D.grad)
    #print(g.quant.grad)
    
    
    
    
    
    
    
    
    
    
    
    

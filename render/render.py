# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:08:10 2026

@author: TogawaSakiko
"""

import torch
from HitoShizuKu import PathTarceForward,ConstructBVH,PathTraceBackward
from camera.camera import PerspectiveCamera

class RenderFunc(torch.autograd.Function):
    @staticmethod 
    def forward(ctx,camera:PerspectiveCamera, 
                texture_id:torch.Tensor,
                faces:torch.Tensor,
                vertices:torch.Tensor,
                uvs:torch.Tensor,
                uv_index:torch.Tensor,
                texture:torch.Tensor,
                f_mtl:torch.Tensor,
                offset:torch.Tensor,
                pMin:torch.Tensor,
                pMax:torch.Tensor,
                triangle_id:torch.Tensor, 
                H:int, 
                W:int, 
                spp:int):
        
        tan_fovx=camera.tan_fovx
        tan_fovy=camera.tan_fovy
        
        cam_to_world=torch.tensor(camera.cam_to_world(),dtype=torch.float32).to(vertices.device)
        
        ray_bias=torch.rand(size=(H,W,spp,2)).to(vertices.device)-0.5
        ray_bias=torch.zeros((H,W,spp,2),dtype=torch.float32).to(vertices.device)
        
        
        image,intersected_tris,res_text_id=PathTarceForward(texture_id,faces,vertices,uvs,uv_index,texture,f_mtl,
                               cam_to_world, offset,pMin,pMax,triangle_id, ray_bias,H, W, tan_fovy,tan_fovx,spp)
        
        ctx.save_for_backward(faces,vertices,uvs, uv_index,texture,cam_to_world,ray_bias,intersected_tris,res_text_id)
        ctx.H=H
        ctx.W=W
        ctx.tan_fovx=tan_fovx
        ctx.tan_fovy=tan_fovy
        ctx.spp=spp
        
        return image
    @staticmethod 
    def backward(ctx,grad_out):
        
        faces,vertices,uvs, uv_index,texture,cam_to_world,ray_bias,intersected_tris,res_text_id=ctx.saved_tensors
        
        H=ctx.H
        W=ctx.W
        tan_fovx=ctx.tan_fovx
        tan_fovy=ctx.tan_fovy
        spp=ctx.spp
        
        d_vertices=torch.zeros_like(vertices)
        d_uv=torch.zeros_like(uvs)
        d_texture=torch.zeros_like(texture)
        
        
        dV,dUV,dT=PathTraceBackward(faces,vertices,uvs, uv_index,texture,cam_to_world,ray_bias,intersected_tris,res_text_id, H, 
                                    W,tan_fovy,tan_fovx,spp,grad_out,d_vertices,d_uv,d_texture)
        
        return None, None,None,dV,dUV,None,dT,None,None,None,None,None,None,None,None
    
def Render(camera:PerspectiveCamera, 
            texture_id:torch.Tensor,
            faces:torch.Tensor,
            vertices:torch.Tensor,
            uvs:torch.Tensor,
            uv_index:torch.Tensor,
            texture:torch.Tensor,
            f_mtl:torch.Tensor,
            offset:torch.Tensor,
            pMin:torch.Tensor,
            pMax:torch.Tensor,
            triangle_id:torch.Tensor, 
            H:int, 
            W:int, 
            spp:int):
    return RenderFunc.apply(camera,
                texture_id,
                faces,
                vertices,
                uvs,
                uv_index,
                texture,
                f_mtl,
                offset,
                pMin,
                pMax,
                triangle_id, 
                H, 
                W, 
                spp)
    
def BuildBVH(faces:torch.Tensor,vertices:torch.Tensor):
    
    if faces.is_cpu or vertices.is_cpu:
        raise RuntimeError(f"BVH构建仅支持cuda运算, 当前面索引位于 {faces.device}, 顶点坐标位于{vertices.device}")
    
    offset,pMin,pMax,triangle_id=ConstructBVH(faces,vertices)
    offset.requires_grad=False
    pMin.requires_grad=False
    pMax.requires_grad=False
    triangle_id.requires_grad=False
    
    return offset,pMin,pMax,triangle_id
        
        
        
        
        

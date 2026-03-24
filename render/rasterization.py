# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:22:13 2026

@author: Togawa Sakiko
"""

import torch

def naive_rasterize(height,width,depth, position,features):
    '''
        未考虑性能的光栅成像函数
        
        输入：
            height:图像的高度
            width：图像的宽度
            depth：[F,3] dtype=torch.float 每个三角面顶点的深度
            position: [F,3,2] dtype=torch.float 每个三角面的顶点在视平面上的二维坐标，值介于[-1,1]之间，是归一化后的坐标
            
            features: [F,3,dim] dtype=torch.float 每个三角面顶点的特征，例如rgb值，透明度等等
        
        输出：
            image: [height,width,dim] dtype=torch.float 输出成像后的图像
            rendered_idx : [height,width] dtype=torch.long 输出每个像素是由哪个面成像的，如果一个像素不和任何一个面相交，那么输出-1.
            
        步骤：
            step 1：如何把三角面在视平面上的坐标从归一化转换到 [height,width]的画布上
                Hint:先关于x轴对称，然后平移(1,1) 然后乘以(height/2,width/2)
            
            
    '''
    
    position[:,:,1]=-1*position[:,:,1]
    position=position+1
    position=torch.Tensor([[[height/2,width/2]]],device=position.device)*position
    
    
    
    #下面开始粗暴渲染
    
    
def test_single_triangle():
    height=25
    width=25
    
    depth=torch.Tensor([[0.3230,0.3230,0.3230]])
    position=torch.Tensor([[[-0.3062,0.1768],[0.3062,0.1768],[0.0000,-0.3536]]])
    features=[[[255,0,0],[0,255,0],[0,0,255]]]
    naive_rasterize(height, width, depth, position, features)
    
if __name__=="__main__":
    test_single_triangle()
    
        
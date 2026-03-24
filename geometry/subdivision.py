# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:02:14 2026

@author: Togawa Sakiko
"""

import torch

def adjacency_matrix(vertices:torch.Tensor, edges=torch.LongTensor):
    '''
        返回一个 [N,N]的tensor，表示每个顶点和哪些顶点相邻，i.e. 返回这个mesh的邻接矩阵
        
        输入：
        
            vertices：[N,3] 顶点坐标，dtype=torch.float
            edges：[E,2] 面索引， dtype=torch.int
        
    '''
    N=vertices.shape[0]
    matrix=torch.zeros(size=(N*N,),device=vertices.device).long()
    
    e=N*edges[:,0]+edges[:,1]
    
    matrix[e]=torch.ones(size=(edges.shape[0],),device=edges.device).long()
    matrix=matrix.reshape((N,N))
    return matrix+matrix.T
    
    

def subidivision_trianglemesh_one(vertices:torch.Tensor, faces=torch.LongTensor,alpha=None):
    '''
        进行一次三角网格的循环细分，同样要保持输出顶点坐标对于输入顶点坐标的可微性。
        
        输入：
        
            vertices：[N,3] 顶点坐标，dtype=torch.float
            faces：[F,3] 面索引， dtype=torch.int
            alpha (Optional)：[N,] 每个顶点处的平滑系数，如果没给，默认用参考文献里面的
            
        输出：
            
        new_vertices: [N_{1},3] 细分后的顶点坐标， dtype=torch.float
        new_faces: [F_{1},3] 细分后的面索引， dtype=torch.float
        
        步骤：
            step 1：每个三角面按中点分成四块，完成新面的索引。新产生的顶点记为奇数顶点，原来的顶点称为偶数顶点。
            step 2：调整所有顶点的位置，对于奇数顶点，它有两个相邻的偶数顶点和两个相对的偶数顶点，从而可更新坐标为
                            v_{1}=1/8 * (两个相对的偶数顶点的和)+3/8 *(两个相邻的偶数顶点的和)
                            
                    对于偶数顶点，记k为和它相邻的偶数顶点的数目 其中alpha为这个顶点的平滑系数
                            v_{1}=v*(1-k*alpha)+ k*alpha*(与它相邻的偶数顶点的均值)
        
    '''
    if alpha==None:
        alpha=(0.375+0.25*torch.cos(2*torch.pi/torch.arange(1, vertices.shape[0]+1, step=1,device=vertices.device)))**2+0.375
    
    '''
        为了完成新顶点和新面索引的生成，我们需要获取每个面的三个顶点坐标
    '''
    N=vertices.shape[0]
    F=faces.shape[0]
    #一个形如[F,3,3]的Tensor, [i,j,:]代表第i+1个面的第j+1个顶点的三维坐标
    face_coordinate=torch.gather(vertices,dim=0,index=faces.reshape((F*3,1)).broadcast_to((F*3,3))).reshape(shape=(F,3,3))
    v1=face_coordinate[:,0,:]
    v2=face_coordinate[:,1,:]
    v3=face_coordinate[:,2,:]
    #新顶点的初始化是通过旧顶点的中点实现的
    v12=(v1+v2)*0.375
    v13=(v1+v3)*0.375
    v23=(v2+v3)*0.375
    #然后三角面的四分, 注意到v1,v12,v13是一群三角面, v2,v12,v23是一群三角面, v3,v13,v23是一群三角面, v12,v13,v23是一群三角面, 共有四组
    v1_idx=faces[:,0]
    v2_idx=faces[:,1]
    v3_idx=faces[:,2]
    v12_idx=torch.arange(N,N+F , step=1,device=vertices.device)
    v13_idx=torch.arange(N+F,N+2*F , step=1,device=vertices.device)
    v23_idx=torch.arange(N+2*F,N+3*F , step=1,device=vertices.device)
    #我们暂不去除重复顶点
    
    f1=torch.stack([v1_idx,v12_idx,v13_idx],dim=1)
    f2=torch.stack([v2_idx,v12_idx,v23_idx],dim=1)
    f3=torch.stack([v3_idx,v23_idx,v13_idx],dim=1)
    f4=torch.stack([v23_idx,v12_idx,v13_idx],dim=1)
    new_faces=torch.cat([f1,f2,f3,f4],dim=0)
    #然后给顶点去重, 我们的规则是, 偶数顶点在前, 给奇数顶点去重之后(注意到重复的顶点只能是奇数顶点)，和偶数顶点拼接形成新顶点
    e12=torch.stack([v1_idx,v2_idx],dim=-1)
    e13=torch.stack([v1_idx,v3_idx],dim=-1)
    e23=torch.stack([v2_idx,v3_idx],dim=-1)
    new_v=torch.concat([e12,e13,e23],dim=0).sort(dim=-1)[0]
    new_v,inverse=torch.unique(new_v,sorted=True,return_inverse=True,dim=0)
    #去重的同时把对顶点计算出来
    oppose_12=v3_idx
    oppose_13=v2_idx
    oppose_23=v1_idx
    
    oppose=torch.cat([oppose_12,oppose_13,oppose_23],dim=0)
    
    _,ind=inverse.sort(dim=0)
    
    oppose=oppose[ind].reshape((-1,2))
    oppose_1=oppose[:,0].unsqueeze(-1)
    oppose_2=oppose[:,1].unsqueeze(-1)
    
    #计算new_alpha，这也由同样的插值得到
    new_alpha=torch.gather(alpha,dim=0,index=new_v[:,0])+torch.gather(alpha,dim=0,index=new_v[:,1])
    new_alpha=0.375*new_alpha
    
    oppo_alpha=torch.gather(alpha,dim=0,index=oppose[:,0])+torch.gather(alpha,dim=0,index=oppose[:,1])
    new_alpha+=0.125*oppo_alpha
    new_alpha=torch.concat([alpha,new_alpha],dim=0)
    
    #计算相邻的顶点
    new_v=torch.gather(vertices, dim=0, index=new_v[:,0].unsqueeze(-1).broadcast_to((new_v.shape[0],3)))+torch.gather(vertices, dim=0, index=new_v[:,1].unsqueeze(-1).broadcast_to((new_v.shape[0],3)))
    new_v=0.375*new_v
    
    oppo_v=torch.gather(vertices, dim=0, index=oppose_1.broadcast_to((oppose_1.shape[0],3)))+torch.gather(vertices, dim=0, index=oppose_2.broadcast_to((oppose_2.shape[0],3)))
    new_v+=0.125*oppo_v
    
    
    #给面去重，构造一个inverse作为去重前的index到去重后的index的映射，然后直接索引inverse[new_faces]就得到结果
    inverse=torch.cat([torch.arange(0,N,1,device=vertices.device),inverse+N],dim=-1)
    new_faces=inverse[new_faces]
    
    '''
        下面按照文章中的公式更新旧顶点，这只需考虑邻接矩阵
    '''
    e=torch.concat([e12,e13,e23],dim=0)
    e=torch.unique(e,sorted=False,dim=0)
    #邻接矩阵(不考虑自己和自己相连)必定是对角阵
    m=adjacency_matrix(vertices,e)
    k=m.sum(dim=-1,keepdim=True)
    
    adj_v=vertices.unsqueeze(0).broadcast_to((N,N,3))*m.unsqueeze(-1).broadcast_to(*m.shape,3)
    
    vertices=(1.0-k*alpha.unsqueeze(-1))*vertices+alpha.unsqueeze(-1)*adj_v.sum(dim=-2)
    
    new_vertices=torch.concat([vertices,new_v],dim=0)

    
    return new_vertices,new_faces, new_alpha

def subidivision_trianglemesh_iters(vertices,faces,alpha=None,iters=1):
    for _ in range(0,iters):
        vertices,faces,alpha=subidivision_trianglemesh_one(vertices,faces,alpha)
    return vertices,faces



if __name__=="__main__":
    device='cpu'

    vertices=torch.Tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],device=device).float()
    faces=torch.Tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],device=device).long()
    alpha=torch.Tensor([0,0,0,0]).float()
    
    #edge_face(vertices,faces)
    #subidivision_trianglemesh_one(vertices,faces,alpha)

    v,f=subidivision_trianglemesh_iters(vertices, faces,alpha,iters=3)






















































# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:53:18 2026

@author: Togawa Sakiko
"""

import torch

'''
    
    这个文件包含用torch实现的行进四面体和体积细分算法
    
    参考文献：
    
        Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis
    
        section 3.1.2
        section 3.1.3
'''


def marching_tetrahedra(vertices:torch.Tensor,tet:torch.LongTensor, sdf:torch.Tensor, returen_tet_idx=False):
    '''
        从一个固定的四面体网格tet中提取网格面, 要求提出的曲面顶点关于输入顶点和面是可微的.    
    
        输入：
            vertices: [N,3] 顶点位置, dtype=torch.float
            tet: [K,4] 每个四面体对应的顶点坐标, dtype=torch.int
            sdf: [N,] 每个顶点处的signed distance function的取值, dtype=torch.float
            
            return_tet_idx: 可选 默认为否. 是否返回提取出的面从属的四面体的序号.
            
        输出:
            1. 从tet中提取出的曲面的顶点
            2. 从tet中提取出的曲面的面
            3. 每个面从属的tet中的四面体的序号
    '''
    
    '''一个可行的思路是先逐个计算每个四面体中是否能导出面, 如果能导出的话把顶点坐标记录下来, 然后通过torch.gather和向量的线性组合运算完成梯度的传播
        
        step 1 :
            
            对tet中的每一条边, 观察是否有曲面的顶点. i.e 边的端点处sdf值是否异号. 如异号, 则通过线性插值找到sdf为0的点, 这就是曲面的顶点.
            遍历每一条边之后这些点就是重建曲面的顶点.
        
        step 2 :
            
            遍历每个tet, 观察这些顶点是如何形成面的, 把面的坐标构建出来.
    '''
    
    '''
        更具体地, 为了不重复的获取每个边, 我们可以把每个边按照顶点顺序排序 e=(v1,v2), 然后放入一个集合中. 每个tet有六个边
        
    '''
    edges=set()
    f1_index=[]
    f2_index=[]
    edge_to_tet=dict()
    for i in range(0,tet.shape[0]):
        # Ti是具有四个元素的列表, 总共组成6个顶点
        
        Ti=tet[i].cpu().tolist()
        for k in range(0,3):
            for j in range(k+1,4):
                edges.add(tuple(sorted([Ti[k],Ti[j]])))
                #构建边到四面体之间的对应, 因为一个边可能从属于多个四面体
                if tuple(sorted([Ti[k],Ti[j]])) in edge_to_tet:
                    edge_to_tet[tuple(sorted([Ti[k],Ti[j]]))].append(i)
                else:
                    edge_to_tet[tuple(sorted([Ti[k],Ti[j]]))]=list([i])
                
    #序号小的边在前. 
    edges=sorted(edges)
    edge_to_v=dict()
    tet_index={key : [] for key in [ i for i in range(0,tet.shape[0])]}
    for e in edges:
        v1,v2=e
        f1,f2= sdf[v1].cpu().item(), sdf[v2].cpu().item()
        ''' 如果两个顶点的sdf异号, 则必有新顶点, 考虑插值 
                tf1+(1-t)f2=0    t=-f2/(f1-f2)
        '''                    
        if f1*f2<0:
            f1_index.append(v1)
            f2_index.append(v2)
            #构建边到新顶点的对应
            edge_to_v[e]=len(f1_index)-1
            #构建四面体到新顶点的对应, i.e 每个四面体包含了那些新顶点.
            for item in edge_to_tet[e]:
                tet_index[item].append(len(f1_index)-1)
    
    if len(f1_index)==0:
        return None,None,None
    
    f1_index=torch.LongTensor(f1_index,device=sdf.device)
    f2_index=torch.LongTensor(f2_index,device=sdf.device)
    
    f1=torch.gather(sdf,dim=-1,index=f1_index)
    f2=torch.gather(sdf,dim=-1,index=f2_index)
    t=-f2/(f1-f2)
    ''' 
        这样就能计算新顶点的位置:
            v_{new}=tv1+(1-t)v2
    '''
    v1=torch.gather(vertices, dim=0, index=f1_index.unsqueeze(-1).broadcast_to(f1_index.shape[0],3))
    v2=torch.gather(vertices, dim=0, index=f2_index.unsqueeze(-1).broadcast_to(f2_index.shape[0],3))
    new_vertices= t.unsqueeze(-1)*v1+(1.0-t.unsqueeze(-1))*v2
    
    #print(new_vertices)
    
    
    '''剩余的步骤就是确定面的位置, 我们注意到面的表示是整数Tensor, 所以必定是关于input不可微的, 因此在构建面的时候不需要考虑保持梯度的传播'''
    
    new_faces=[]
    tet_idx=[]
    for k,v in tet_index.items():
        #要么是四边面, 要么是三边面
        if len(v)==4:
            T=tet[k]
            #提取这个四面体的四个顶点的sdf
            f=torch.gather(sdf,dim=0,index=T)
            positive=torch.where(f>0)[0].cpu().tolist()
            negative=torch.where(f<0)[0].cpu().tolist()
            positive=T[positive].cpu().tolist()
            negative=T[negative].cpu().tolist()
            #两个positive的顶点
            
            v2=tuple(sorted([positive[0],negative[0]]))
            v3=tuple(sorted([positive[1],negative[1]]))
            
            v4=tuple(sorted([positive[1],negative[0]]))
            v5=tuple(sorted([positive[0],negative[1]]))
            #组成两个三角面
            f1=[edge_to_v[v2],edge_to_v[v3],edge_to_v[v4]]
            f2=[edge_to_v[v2],edge_to_v[v3],edge_to_v[v5]]
            new_faces.append(f1)
            tet_idx.append(k)
            new_faces.append(f2)
            tet_idx.append(k)
        elif len(v)==3:
            T=tet[k]
            #提取这个四面体的四个顶点的sdf
            f=torch.gather(sdf,dim=0,index=T)
            positive=torch.where(f>0)[0].cpu().tolist()
            negative=torch.where(f<0)[0].cpu().tolist()
            positive=T[positive].cpu().tolist()
            negative=T[negative].cpu().tolist()
            #两个positive的顶点
            if len(positive)==3:
                v2=tuple(sorted([positive[0],negative[0]]))
                v3=tuple(sorted([positive[1],negative[0]]))
                v4=tuple(sorted([positive[2],negative[0]]))
            elif len(negative)==3:
                v2=tuple(sorted([positive[0],negative[0]]))
                v3=tuple(sorted([positive[0],negative[1]]))
                v4=tuple(sorted([positive[0],negative[2]]))

            #组成两个三角面
            f1=[edge_to_v[v2],edge_to_v[v3],edge_to_v[v4]]
            new_faces.append(f1)
            tet_idx.append(k)
    new_faces=torch.Tensor(new_faces,device=new_vertices.device).long()
    
    if returen_tet_idx:
        return new_vertices ,new_faces,tet_idx
    else:
        return new_vertices ,new_faces,None
    
def marching_tetrahedra_vertorized(vertices:torch.Tensor,tet:torch.LongTensor, sdf:torch.Tensor, returen_tet_idx=False):
    '''
        顾名思义, 比起上面的原始版本, 该版本去掉了所有for循环, 是更高效的实现
        
        第一个循环在建立边, 对于这一步, 我们可以把每个tet的四个顶点通过组合数公式获取六种情况 变成[K,6,2], 考虑torch.gather
        然后变成[6K,2], 6K里面大概率有重复的, 考虑torch.unique
        
        第二个循环构建了有新顶点的边, 可以考虑torch.gather获取f在每个边的左顶点的值和右顶点的值, 然后torch.where取出异号的index, 这些index上会出现新的顶点.
        
        最后一个循环确定了面的位置
    
    '''
    K=tet.shape[0]
    #[1,6,2]
    ind=torch.combinations(input=torch.arange(0, 4, step=1,device=vertices.device),r=2).unsqueeze(0)
    ind=ind.broadcast_to((K,6,2))
    
    edges=torch.stack([torch.gather(tet, dim=-1, index=ind[:,:,0]), torch.gather(tet, dim=-1, index=ind[:,:,1])],dim=-1)
    edges=edges.reshape((6*K,2)).sort(dim=-1)[0]
    edges,inverse=torch.unique(edges,sorted=True,return_inverse=True,dim=0)
    inverse=inverse.reshape((K,6))
    
    #然后优化第二个循环
    f1=torch.gather(input=sdf, dim=0, index=edges[:,0])
    f2=torch.gather(input=sdf, dim=0, index=edges[:,1])
    
    ind=torch.where(f1*f2<0)
    ind=ind[0]
    
    #edges[ind]就是对应会产生新顶点的边. 
    f1=torch.gather(input=sdf,dim=0,index=edges[ind][:,0])
    f2=torch.gather(input=sdf,dim=0,index=edges[ind][:,1])
    v1=torch.gather(input=vertices,dim=0,index=edges[ind][:,0].unsqueeze(-1).broadcast_to(ind.shape[0],3))
    v2=torch.gather(input=vertices,dim=0,index=edges[ind][:,1].unsqueeze(-1).broadcast_to(ind.shape[0],3))
    t=-f2/(f1-f2)
    t=t.unsqueeze(-1)
    
    new_vertices=t*v1+(1.0-t)*v2
    #每个边对应的新顶点的编号
    edge_to_v=-1*torch.ones(size=(edges.shape[0],),device=vertices.device,dtype=torch.long)
    edge_to_v[ind]=torch.arange(0, ind.shape[0], step=1,device=vertices.device)
    #构建掩码矩阵 每个tet的六个边 如果边上有新顶点则为1, 否则为0
    mask= edge_to_v[inverse]
    mask=torch.where(mask==-1,False,True).long()
    
    #分辨获取包含三角面和四边面的四面体的编号
    triangle_tet_index=torch.where(mask.sum(dim=-1)==3)[0]
    rectangle_tet_index=torch.where(mask.sum(dim=-1)==4)[0]
    #为添加三角面很容易, 问题是如何添加四边面, 怎么把四边面划分成三角面？ 这需要找到四边面的对角线.
    tet4=torch.gather(tet,dim=0,index=rectangle_tet_index.unsqueeze(-1).broadcast_to((rectangle_tet_index.shape[0],4)))
    f=sdf[tet4]
    
    #这些f必定两个正两个负
    positive=torch.where(f>0)[1]
    positive=torch.reshape(positive, (-1,2))
    #positive_edge=torch.gather(tet4,dim=-1,index=positive)
    
    negative=torch.where(f<0)[1]
    negative=negative.reshape((-1,2))
    #negative_edge=torch.gather(tet4, dim=-1, index=negative)
    
    look_up=torch.Tensor([0,1,2,3,4,-1,5],device=vertices.device).long()
    
    e1=torch.stack([positive[:,0],negative[:,0]],dim=-1).sort(dim=-1)[0]
    e2=torch.stack([positive[:,1],negative[:,1]],dim=-1).sort(dim=-1)[0]
    
    e3=torch.stack([positive[:,1],negative[:,0]],dim=-1).sort(dim=-1)[0]
    e4=torch.stack([positive[:,0],negative[:,1]],dim=-1).sort(dim=-1)[0]
    
    e1=2*e1[:,0]+e1[:,1]-1
    e1=look_up[e1]
    e2=2*e2[:,0]+e2[:,1]-1
    e2=look_up[e2]
    e3=2*e3[:,0]+e3[:,1]-1
    e3=look_up[e3]
    e4=2*e4[:,0]+e4[:,1]-1
    
    e1=look_up[e1.unsqueeze(-1)]
    e2=look_up[e2.unsqueeze(-1)]
    e3=look_up[e3.unsqueeze(-1)]
    e4=look_up[e4.unsqueeze(-1)]
    
    #把tet4变成新边索引，注意到新顶点的序号和新边的序号一样，因为新边上有且仅有一个新顶点
    tet4=torch.gather(inverse,dim=0,index=rectangle_tet_index.unsqueeze(-1).broadcast_to((rectangle_tet_index.shape[0],6)))
    e1=torch.gather(tet4,dim=-1,index=e1)
    e2=torch.gather(tet4,dim=-1,index=e2)
    e3=torch.gather(tet4,dim=-1,index=e3)
    e4=torch.gather(tet4,dim=-1,index=e4)
    #e1,e2,e3一组, e1,e2,e4一组
    f1=torch.concat([e1,e2,e3],dim=-1)
    f2=torch.concat([e1,e2,e4],dim=-1)
    
    ##然后获取三角面
    mask3=torch.gather(mask, dim=0, index=triangle_tet_index.unsqueeze(-1).broadcast_to((triangle_tet_index.shape[0],6)))
    tet3=torch.gather(inverse,dim=0,index=triangle_tet_index.unsqueeze(-1).broadcast_to((triangle_tet_index.shape[0],6)))
    
    mask3=torch.where(mask3==1)[1]
    f3=torch.gather(tet3,dim=-1,index=mask3.reshape(-1,3))
    sox=torch.arange(0,edges.shape[0],step=1,device=vertices.device)
    sox[ind]=torch.arange(0, ind.shape[0], step=1,device=vertices.device)
    f3=sox[f3]
    f1=sox[f1]
    f2=sox[f2]
    new_faces=torch.concat([f1,f2,f3],dim=0)
    
    if returen_tet_idx:
        #必须要谨慎处理idx, 注意到f1和f2的idx是一样的。
        idx3=triangle_tet_index
        idx1=rectangle_tet_index
        tet_idx=torch.concat([idx1,idx1,idx3],dim=0)
        return new_vertices,new_faces,tet_idx
    else:
        return new_vertices,new_faces,None
    
def volume_subdivision(vertices:torch.Tensor, tet:torch.LongTensor, features=None):
    ''' 
        对给出的四面体进行一次细分，细分规则是每个四面体的每条边生成一个中点，把这个四面体分成八个小的四面体，如果给出了顶点特征，那么对新顶点直接用平均插值获得新特征
        
        输入：
        
            vertices：[N,3] 顶点坐标，dtype=torch.float
            tet：[K,4] 四面体， dtype=torch.int
            features (Optional)：[N,dim] 每个顶点处的特征，
            
        输出：
        
            new_vertices: [N_{1},3] 新顶点坐标，dtype=torch.float
            new_tet: [K_{1},4] 新的四面体索引，dtype=torch.long
            new_features (Optional): [N_{1},3] 新的顶点特征
            
        步骤：
        
            整体的实现是比较容易的。先构造出新顶点（暂不去重），然后把新四面体的索引构建出来，给顶点去重之后把新四面体的索引映射到去重之后的索引
            
            feature的插值是容易的
    '''
    N=vertices.shape[0]
    K=tet.shape[0]
    
    va=tet[:,0]
    vb=tet[:,1]
    vc=tet[:,2]
    vd=tet[:,3]
    
    vab,vac,vad,vbc,vbd,vcd=torch.split(torch.arange(N, N+6*K, step=1,device=vertices.device),split_size_or_sections=K)
    
    #构建new_tet的索引
    tet1=torch.stack([va,vab,vac,vad],dim=-1)
    tet2=torch.stack([vb,vab,vbc,vbd],dim=-1)
    tet3=torch.stack([vc,vac,vbc,vcd],dim=-1)
    tet4=torch.stack([vd,vad,vbd,vcd],dim=-1)
    
    tet5=torch.stack([vac,vbc,vcd,vbd],dim=-1)
    tet6=torch.stack([vab,vac,vbc,vbd],dim=-1)
    tet7=torch.stack([vad,vac,vbd,vcd],dim=-1)
    tet8=torch.stack([vab,vac,vad,vbd],dim=-1)
    
    new_tet=torch.concat([tet1,tet2,tet3,tet4,tet5,tet6,tet7,tet8],dim=0)
    #print(new_tet)
    
    ind_ab=torch.stack([va,vb],dim=-1).sort(dim=-1)[0]
    ind_ac=torch.stack([va,vc],dim=-1).sort(dim=-1)[0]
    ind_ad=torch.stack([va,vd],dim=-1).sort(dim=-1)[0]
    ind_bc=torch.stack([vb,vc],dim=-1).sort(dim=-1)[0]
    ind_bd=torch.stack([vb,vd],dim=-1).sort(dim=-1)[0]
    ind_cd=torch.stack([vc,vd],dim=-1).sort(dim=-1)[0]
    
    e=torch.concat([ind_ab,ind_ac,ind_ad,ind_bc,ind_bd,ind_cd],dim=0)
    
    e,inverse=torch.unique(e,sorted=True,return_inverse=True,dim=0)

    inverse=torch.concat([torch.arange(0, N, step=1,device=vertices.device),inverse+N],dim=-1)
    
    new_tet=inverse[new_tet]
    
    new_v=torch.gather(vertices,dim=0,index=e[:,0].unsqueeze(-1).broadcast_to((e.shape[0],3)))+torch.gather(vertices,dim=0,index=e[:,1].unsqueeze(-1).broadcast_to((e.shape[0],3)))
    new_v=0.5*new_v
    
    if features!=None:
        new_features=torch.gather(features,dim=0,index=e[:,0].unsqueeze(-1).broadcast_to((e.shape[0],features.shape[-1])))+torch.gather(features,dim=0,index=e[:,1].unsqueeze(-1).broadcast_to((e.shape[0],features.shape[-1])))
        new_features=0.5*new_features
        new_features=torch.concat([features,new_features],dim=0)
    new_vertices=torch.concat([vertices,new_v],dim=0)
    if features==None:
        return new_vertices,new_tet,None
    else:
        return new_vertices,new_tet,new_features
    

if __name__=="__main__":
    device='cpu'
    vertices=torch.Tensor([[0,0,0],[1,0,0,],[0,1,0],[0,0,1],[0,-1,0]]).float()
    tet=torch.Tensor([[0,1,2,3],[0,1,3,4]]).long()
    sdf=torch.Tensor([-1.0,-1.0,0.5,0.5,0.5]).float()
    #marching_tetrahedra_vertorized(vertices, tet, sdf)
    
    vertices=torch.Tensor([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]],device=device).float()
    tet=torch.Tensor([[0,1,3,5],[4,5,0,6],[0,3,2,6],[5,3,6,7],[0,5,3,6]],device=device).long()
    
    sdf=torch.Tensor([[1,1,1,1,1,1,1,1],[1,1,1,1,-1,1,1,1],[1,1,1,1,-1,-1,1,1],[1,1,1,1,-0.5,-0.7,1,1]]).float()
    
    vertices.requires_grad_()
    marching_tetrahedra_vertorized(vertices, tet, sdf[3])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
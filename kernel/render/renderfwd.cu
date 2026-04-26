#include <math_utils.h>
#include <render.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <fstream>
#include <cub/cub.cuh>
#include <sah.h>



namespace cg=cooperative_groups;

RayBuffer RayBuffer::ALLOC(char*& address, size_t num_pixels)
{
	RayBuffer r;
	manual_allocate(address, r.rays, num_pixels, 128);
	r.num_rays = num_pixels;

	return r;

}

Scene Scene::ALLOC(char*& address, size_t num_primitives, size_t num_materials)
{
	Scene s;
	manual_allocate(address, s.materials, num_materials, 128);
	manual_allocate(address, s.primitives, num_primitives, 128);
	s.num_primitives = num_primitives;
	s.num_materials = num_materials;

	return s;
}

BVHBuffer BVHBuffer::ALLOC(char*& address, size_t num_nodes)
{
	BVHBuffer b;
	manual_allocate(address, b.bvhs, num_nodes, 128);
	b.num_nodes=num_nodes;

	return b;
}

InteractionBuffer  InteractionBuffer::ALLOC(char*& address, size_t num_interactions)
{
	InteractionBuffer  i;
	manual_allocate(address, i.interactions, num_interactions, 128);
	i.num_interactions=num_interactions;

	return i;

}


__global__ void SetMaterial(Material* materials,const int* texture_id,int num_materials)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_materials)
	{
		return;
	}
	else
	{
		materials[idx].texture_id=texture_id[idx];

		
	}
	
}

__global__ void constructScene(Primitive* primitives,const int* triangles, const float* vertices, const float* uvs,
								const int* uv_index, const int* f_mtl,const int num_triangles)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_triangles)
	{
		return;
	}
	int index=triangles[3*idx];
	primitives[idx].points[0]= vec3f(vertices[3*index],vertices[3*index+1],vertices[3*index+2]);
	
	index=triangles[3*idx+1];
	primitives[idx].points[1]=vec3f(vertices[3*index],vertices[3*index+1],vertices[3*index+2]);
	
	index=triangles[3*idx+2];
	primitives[idx].points[2]=vec3f(vertices[3*index],vertices[3*index+1],vertices[3*index+2]);
	



	index=uv_index[3*idx];
	primitives[idx].uv[0]=vec2f(uvs[2*index],uvs[2*index+1]);
	index=uv_index[3*idx+1];
	primitives[idx].uv[1]=vec2f(uvs[2*index],uvs[2*index+1]);
	index=uv_index[3*idx+2];
	primitives[idx].uv[2]=vec2f(uvs[2*index],uvs[2*index+1]);
	primitives[idx].material_id=f_mtl[idx];
	

	primitives[idx].BBox=Union(Union(Bounds3(primitives[idx].points[0]),primitives[idx].points[1]),primitives[idx].points[2]);

}

__global__ void RayGeneration(Ray* rays, const float* cam_to_world, const float tan_fovx, 
								const float tan_fovy, const float* ray_bias, const int H, const int W, const int num_rays,const int iter)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=H*W)
	{
		return;
	}
	//我们需要注意的是tan_fovx指的是一半的宽度除以相机到平面的距离
	float w=1.0*tan_fovx;
	float h=1.0*tan_fovy;

	//像素在图片平面上的x y坐标
	float pidx=float(idx%W)+0.5f;
	float pidy=float(idx/W)+0.5f;

	int bias_idx=2*H*W*iter+2*idx;
	float dx=ray_bias[bias_idx];
	float dy=ray_bias[bias_idx+1];

	pidx+=dx;
	pidy+=dy;


	//转化为相机空间的点
	vec3f coor;
	coor.x=pidx/float(W)*2*w-w;
	coor.y=pidy/float(H)*2*h-h;
	coor.z=-1.0;

	//相机空间的点投影到世界空间
	vec3f p=ProjectToWorld(coor,cam_to_world);

	vec3f cam_pos=vec3f(cam_to_world[3],cam_to_world[7],cam_to_world[11] );

	rays[idx].dir=(p-cam_pos).normalize();
	rays[idx].ori=cam_pos;

	rays[idx].pix_id=idx;
	rays[idx].color=vec3f(0.0f,0.0f,0.0f);
	rays[idx].tMax=1e10;

}


__global__ void reconstructBVH(BVHInfo* info,const int num_nodes,const int* offset,const float* pMin,const float* pMax,const int* triangle_id)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_nodes)
	{
		return;
	}

	info[idx].offset=offset[idx];
	info[idx].BBox.pMin =vec3f( pMin[3*idx],pMin[3*idx+1],pMin[3*idx+2]);
	info[idx].BBox.pMax =vec3f( pMax[3*idx],pMax[3*idx+1],pMax[3*idx+2]);

	info[idx].triangle_id=triangle_id[idx];
	if(triangle_id[idx]!=-1)
	{
		info[idx].is_leaf=true;
	}
	else
	{
		info[idx].is_leaf=false;
	}

}

__device__ bool GetRayInsection(Ray ray,BVHInfo* info,Primitive* primitive,int& tri_idx,float& curr_t,float& b0,float& b1)
{
	bool ins=false;
	vec3f invDir=vec3f(1.0f/ray.dir.x,1.0f/ray.dir.y,1.0f/ray.dir.z);
	int DirIsNeg[3]={0,0,0};
	if(invDir.x<0) DirIsNeg[0]=1;
	if(invDir.y<0) DirIsNeg[1]=1;
	if(invDir.z<0) DirIsNeg[2]=1;

	float min_t=1000.0f;

	
	//实际应用中我们很难得到一个超出256层的满二叉树
	int ToVisit[256];
	int top=0;

	ToVisit[top]=0;
	while(top>=0)
	{
		//如果相交了且不是叶子结点 就把子节点放进去
		int ind=ToVisit[top];
		top--;
		if(CheckInsect(info[ind].BBox, ray, invDir, DirIsNeg))
		{
			if( !info[ind].is_leaf  )
			{
				//左节点的序号是当前节点的序号+1 右节点的序号是+offset
				ToVisit[top+1]=ind+info[ind].offset;
				ToVisit[top+2]=ind+1;
				top+=2;
			
			}
			else
			{
				
				//相交且是叶子结点 那就检查三角形相交
				Primitive p=primitive[info[ind].triangle_id];
				float temp0=0.0;
				float temp1=0.0f;
				//注意这不要把b0 b1传进去 否则相交会报错
				if(TriangleRayInsect( p,  ray,curr_t,temp0,temp1))
				{
					ins=true;
					if(curr_t<min_t)
					{
						b0=temp0;
						b1=temp1;
						min_t=curr_t;
						tri_idx=info[ind].triangle_id;
						
					}
				}
			}	
		}
		
	}
	curr_t=min_t;
	return ins;
}



__global__ void GetInsection(Interaction* interactions, Ray* rays,BVHInfo* info,Primitive* primitive, 
								const int num_rays,int* intersected_tris,const int iter,const int spp, const int* f_mtl, const int* texture_id,int* res_text_id)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_rays)
	{
		return;
	}
	int triangle_id=-1;
	float curr_t=1.0f;
	float b0=0.0f;
	float b1=0.0f;
	if(GetRayInsection(rays[idx], info,primitive,triangle_id,curr_t,b0,b1))
	{
		//先采样uv
		float b2=1.0f-b0-b1;
		Primitive p=primitive[triangle_id];
		interactions[idx].intersected=true;
		interactions[idx].point=rays[idx].ori+curr_t*rays[idx].dir;

		interactions[idx].uv=b0*p.uv[0]+b1*p.uv[1]+b2*p.uv[2];

		interactions[idx].ray_idx=idx;
		interactions[idx].material_id=p.material_id;


		intersected_tris[spp*rays[idx].pix_id+iter]=triangle_id;
		res_text_id[spp*rays[idx].pix_id+iter]=texture_id[f_mtl[triangle_id]];


		
	}
	else
	{
		interactions[idx].intersected=false;
		interactions[idx].uv=vec2f(0,0);
		interactions[idx].material_id=-1;
		interactions[idx].ray_idx=-1;

		intersected_tris[spp*rays[idx].pix_id+iter]=-1;
		res_text_id[spp*rays[idx].pix_id+iter]=-1;
	}
}


__global__ void Trace(Interaction* interactions ,Ray* rays, const Material* materials ,const float* texture,
						int texture_H, int texture_W ,const int num_interactions)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_interactions)
	{
		return;
	}
	//相交了就进行光线的传播和光谱累计
	vec3f color=vec3f(0,0,0);
	Interaction inter=interactions[idx];
	if(inter.intersected)
	{
		vec2f pix_uv=inter.uv;
		pix_uv.x*=texture_W;
		pix_uv.y*=texture_H;
		int start_x=0,start_y=0;
		
		int texture_id=materials[inter.material_id].texture_id;
						
		if(texture_id>=0 && BilinearSample( pix_uv, texture_H, texture_W,start_x,start_y))
		{
			//如果有贴图可以双线性采样
			uint32_t text_bias=(materials[inter.material_id].texture_id)*(texture_H*texture_W*3);

			int text_id00=start_y* texture_W+start_x;
			int text_id01=(start_y+1)* texture_W+start_x;
			int text_id10=start_y* texture_W+start_x+1;
			int text_id11=(start_y+1)* texture_W+start_x+1;

			float x0=(start_x+1-pix_uv.x)*texture[text_bias+3*text_id00]+(pix_uv.x-start_x)*texture[text_bias+3*text_id10];
			float y0=(start_x+1-pix_uv.x)*texture[text_bias+3*text_id00+1]+(pix_uv.x-start_x)*texture[text_bias+3*text_id10+1];
			float z0=(start_x+1-pix_uv.x)*texture[text_bias+3*text_id00+2]+(pix_uv.x-start_x)*texture[text_bias+3*text_id10+2];

			float x1=(start_x+1-pix_uv.x)*texture[text_bias+3*text_id01]+(pix_uv.x-start_x)*texture[text_bias+3*text_id11];
			float y1=(start_x+1-pix_uv.x)*texture[text_bias+3*text_id01+1]+(pix_uv.x-start_x)*texture[text_bias+3*text_id11+1];
			float z1=(start_x+1-pix_uv.x)*texture[text_bias+3*text_id01+2]+(pix_uv.x-start_x)*texture[text_bias+3*text_id11+2];

			color.x=(start_y+1-pix_uv.y)*x0+(pix_uv.y-start_y)*x1;
			color.y=(start_y+1-pix_uv.y)*y0+(pix_uv.y-start_y)*y1;
			color.z=(start_y+1-pix_uv.y)*z0+(pix_uv.y-start_y)*z1;
		}
		else
		{
			//没有贴图就采样Kd
			color=vec3f(0,0,0);
		}

		int ray_idx=inter.ray_idx;
		rays[ray_idx].color=color;


	}
}

__global__ void shadeKernel(Ray* rays,  float* res,const int num_rays,int spp)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_rays)
	{
		return;
	}
	int pid=rays[idx].pix_id;

	
	
	res[3*pid]+=rays[idx].color.x/spp;
	res[3*pid+1]+=rays[idx].color.y/spp;
	res[3*pid+2]+=rays[idx].color.z/spp;


}


void PathTraceFWD(
	std::function<char*(size_t)> SceneInfo,
	std::function<char*(size_t)> RayInfo,
	std::function<char*(size_t)> BBoxInfo,
	std::function<char*(size_t)> InteractionInfo,
	const int* triangles,
	const float* vertices,
	const float* uvs,
	const int* uv_index,
	const float* texture,
	const int* f_mtl,
	const float* cam_to_world,
	const int* offset,      
	const float* pMin,
	const float* pMax,
	const int* triangle_id,
	const float* ray_bias,
	const float tan_fovy,
	const float tan_fovx,
	int H,
	int W,
	const int* texture_id,
	const int num_triangles,
	const int num_uvs,
	const int num_vertices,
	const int num_materials,
	const int num_nodes,
	const int texture_H,
	const int texture_W,
	const int num_spp,
	float* res,
	int* intersected_tris,
	int* res_text_id)
{
	//先给场景分配足够的空间
	size_t chunk_size=PrecomputeSceneBias(num_triangles,num_materials);
	char* chunk_ptr=SceneInfo(chunk_size);
	Scene scene=Scene::ALLOC(chunk_ptr, num_triangles,num_materials);
	
	dim3 block_dim(BLOCK_X,BLOCK_Y);
	dim3 grid_dim((W+BLOCK_X-1)/BLOCK_X,(H+BLOCK_Y-1)/BLOCK_Y);

	cudaError_t err;

	//step1:为材质和片元赋值
	SetMaterial<<<(num_materials+255)/256,256>>>(scene.materials,texture_id,num_materials);
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("材质初始化失败\n");
	}
	cudaDeviceSynchronize();

	constructScene<<<(num_triangles+255)/256,256>>>(scene.primitives,triangles, vertices, uvs, uv_index,f_mtl,num_triangles);
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("片元初始化失败\n");
	}

	//为光线分配空间
	size_t ray_chunk_size=PrecomputeRayBufferBias(H*W);
	char* ray_chunk_ptr=RayInfo(ray_chunk_size);
	RayBuffer raybuffer=RayBuffer::ALLOC(ray_chunk_ptr, H*W);

	int num_rays=H*W;

	//step2 恢复BVH
	size_t bvh_chunk_size=PrecomputeBVHBufferBias(num_nodes);
	char* bvh_chunk_ptr=BBoxInfo(bvh_chunk_size);
	BVHBuffer b=BVHBuffer::ALLOC(bvh_chunk_ptr, num_nodes);

	reconstructBVH<<<(num_nodes+255)/256,256>>>(b.bvhs, num_nodes, offset,pMin,pMax,triangle_id);
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("SAH主机到设备数据拷贝失败\n");
	}
	
	//step3 准备interaction
	size_t interact_chunk_size=PrecomputeInteractionBufferBias(num_rays);
	char* interact_ptr=InteractionInfo(interact_chunk_size);
	InteractionBuffer inter=InteractionBuffer::ALLOC(interact_ptr,num_rays);

	int i=0,j=0;
	int spp=num_spp;
	for(j=0;j<spp;j++)
	{
		//step4: 生成光线
		
		RayGeneration<<<grid_dim,block_dim>>>(raybuffer.rays,cam_to_world, tan_fovx, 
									tan_fovy, ray_bias,H,  W,  num_rays,j);
		cudaDeviceSynchronize();
		err=cudaGetLastError();
		if(err!=cudaSuccess)
		{
			printf("光线生成失败\n");
		}
		//step5 打光相交一次看结果 
		GetInsection<<<(num_rays+255)/256,256>>>(inter.interactions, raybuffer.rays,b.bvhs,scene.primitives,num_rays,intersected_tris,j,spp,
													f_mtl,  texture_id,res_text_id);
		err=cudaGetLastError();
		if(err!=cudaSuccess)
		{
			printf("光线求交失败\n");
		}

		//step6 对光线积累光谱
		int num_interactions=num_rays;
		Trace<<<(num_rays+255)/256,256>>>(inter.interactions ,raybuffer.rays, scene.materials ,texture, texture_H, texture_W , num_interactions);
		cudaDeviceSynchronize();
		err=cudaGetLastError();
		if(err!=cudaSuccess)
		{
			printf("光线弹射失败\n");
		}
		

		//final step 着色
		shadeKernel<<<(num_rays+255)/256,256>>>(raybuffer.rays,  res,num_rays, spp);
		err=cudaGetLastError();
		if(err!=cudaSuccess)
		{
			printf("着色失败\n");
		}

	}
	


	

	
}


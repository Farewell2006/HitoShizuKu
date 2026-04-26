#include <math_utils.h>
#include <render.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <fstream>
#include <cub/cub.cuh>

namespace cg=cooperative_groups;

__device__ Ray generateRay(int pix_id,int iter,const float* cam_to_world, vec2f bias, const float tan_fovx, const float tan_fovy, const int H, const int W)
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
	float dx=bias.x;
	float dy=bias.y;

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

	Ray r;
	r.dir=(p-cam_pos).normalize();
	r.ori=cam_pos;

	r.pix_id=pix_id;
	r.color=vec3f(0.0f,0.0f,0.0f);
	r.tMax=1e10;

	return r;
}


__global__ void ComputeUV(
						int* intersected_tris,
						const int* triangles,
						const float* vertices,
						const float* uvs,
						const int* uv_index,
						const float* texture,
						const int* texture_id,
						const float* ray_bias,
						const float* cam_to_world,
						const int H, const int W,
						const int texture_H,const int texture_W,
						const float tan_fovx, const float tan_fovy,
						const int iter,
						const int num_spp,
						const float* dO,
						float* db0,
						float* db1,
						float* d_texture,
						float* d_uv)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=H*W)
	{
		return;
	}

	int tri_id=intersected_tris[num_spp*idx+iter];
	if(tri_id<0)
	{
		return;
	}

	int bias_idx=2*H*W*iter+2*idx;
	vec2f bias=vec2f(ray_bias[bias_idx],ray_bias[bias_idx+1]);
	Ray ray=generateRay(idx, iter,cam_to_world, bias,tan_fovx, tan_fovy,  H,  W);


	//只需要顶点和uv信息
	Primitive p;
	int ind=triangles[3*tri_id];
	p.points[0]=vec3f(vertices[3*ind],vertices[3*ind+1],vertices[3*ind+2]);
	ind=triangles[3*tri_id+1];
	p.points[1]=vec3f(vertices[3*ind],vertices[3*ind+1],vertices[3*ind+2]);
	ind=triangles[3*tri_id+2];
	p.points[2]=vec3f(vertices[3*ind],vertices[3*ind+1],vertices[3*ind+2]);

	int index=uv_index[3*tri_id];
	p.uv[0]=vec2f(uvs[2*index],uvs[2*index+1]);
	index=uv_index[3*tri_id+1];
	p.uv[1]=vec2f(uvs[2*index],uvs[2*index+1]);
	index=uv_index[3*tri_id+2];
	p.uv[2]=vec2f(uvs[2*index],uvs[2*index+1]);


	float time=0.0f;
	float b0=0.0f;
	float b1=0.0f;
	TriangleRayInsect(p, ray, time, b0,b1);
	float b2=1.0f-b0-b1;

	vec2f pix_uv=b0*p.uv[0]+b1*p.uv[1]+b2*p.uv[2];
	

	pix_uv.x*=texture_W;
	pix_uv.y*=texture_H;
	int start_x=0,start_y=0;
		
	int texture_idx=texture_id[num_spp*idx+iter];
	if(texture_idx>=0 && BilinearSample( pix_uv, texture_H, texture_W,start_x,start_y))
	{
		//如果有贴图可以双线性采样
		uint32_t text_bias=texture_idx*(texture_H*texture_W*3);

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

		

		float duv_x=0.0f;
		vec3f dcolor=vec3f(dO[3*idx],dO[3*idx+1],dO[3*idx+2]);
		vec2f dL_duv=vec2f(0,0);

		//输出对于目前像素的uv的偏微分 不要忘记乘以贴图宽高
		dL_duv.x=dcolor.x*((start_y+1-pix_uv.y)*(texture[text_bias+3*text_id10]-texture[text_bias+3*text_id00])+(pix_uv.y-start_y)*(texture[text_bias+3*text_id11]-texture[text_bias+3*text_id01]));
		dL_duv.x+=dcolor.y*((start_y+1-pix_uv.y)*(texture[text_bias+3*text_id10+1]-texture[text_bias+3*text_id00+1])+(pix_uv.y-start_y)*(texture[text_bias+3*text_id11+1]-texture[text_bias+3*text_id01+1]));
		dL_duv.x+=dcolor.z*((start_y+1-pix_uv.y)*(texture[text_bias+3*text_id10+2]-texture[text_bias+3*text_id00+2])+(pix_uv.y-start_y)*(texture[text_bias+3*text_id11+2]-texture[text_bias+3*text_id01+2]));
		dL_duv.x*=texture_W;

		dL_duv.y=(dcolor.x*(x1-x0)+dcolor.y*(y1-y0)+dcolor.z*(z1-z0))*texture_H;

		//计算三个顶点uv的梯度 当前像素的uv是由三顶点插值而来的
		index=uv_index[3*tri_id];
		atomicAdd(&(d_uv[2*index]), dL_duv.x*b0 /num_spp);
		atomicAdd(&(d_uv[2*index+1]), dL_duv.y*b0 /num_spp);

		index=uv_index[3*tri_id+1];
		atomicAdd(&(d_uv[2*index]), dL_duv.x*b1/num_spp );
		atomicAdd(&(d_uv[2*index+1]), dL_duv.y*b1/num_spp);

		
		index=uv_index[3*tri_id+2];
		atomicAdd(&(d_uv[2*index]), dL_duv.x*b2 /num_spp);
		atomicAdd(&(d_uv[2*index+1]), dL_duv.y*b2/num_spp);


		//然后计算该像素关于b0 b1的偏微分 这会影响顶点的梯度
		db0[idx]=dL_duv.x*(p.uv[0].x-p.uv[2].x)+dL_duv.y*(p.uv[0].y-p.uv[2].y);
		db1[idx]=dL_duv.x*(p.uv[1].x-p.uv[2].x)+dL_duv.y*(p.uv[1].y-p.uv[2].y);



		//计算关于纹理的梯度

		atomicAdd(&(d_texture[text_bias+3*text_id00]),dcolor.x*(start_y+1-pix_uv.y)*(start_x+1-pix_uv.x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id00+1]),dcolor.y*(start_y+1-pix_uv.y)*(start_x+1-pix_uv.x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id00+2]),dcolor.z*(start_y+1-pix_uv.y)*(start_x+1-pix_uv.x)/num_spp);

		atomicAdd(&(d_texture[text_bias+3*text_id10]),dcolor.x*(start_y+1-pix_uv.y)*(pix_uv.x-start_x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id10+1]),dcolor.y*(start_y+1-pix_uv.y)*(pix_uv.x-start_x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id10+2]),dcolor.z*(start_y+1-pix_uv.y)*(pix_uv.x-start_x)/num_spp);

		atomicAdd(&(d_texture[text_bias+3*text_id01]),dcolor.x*(pix_uv.y-start_y)*(start_x+1-pix_uv.x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id01+1]),dcolor.y*(pix_uv.y-start_y)*(start_x+1-pix_uv.x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id01+2]),dcolor.z*(pix_uv.y-start_y)*(start_x+1-pix_uv.x)/num_spp);

		atomicAdd(&(d_texture[text_bias+3*text_id11]),dcolor.x*(pix_uv.y-start_y)*(pix_uv.x-start_x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id11+1]),dcolor.y*(pix_uv.y-start_y)*(pix_uv.x-start_x)/num_spp);
		atomicAdd(&(d_texture[text_bias+3*text_id11+2]),dcolor.z*(pix_uv.y-start_y)*(pix_uv.x-start_x)/num_spp);
		


	}
		
}


__global__ void ComputedV(int* intersected_tris,
						const int* triangles,const float* vertices,const float* ray_bias,
						const float* cam_to_world,
						const int H, const int W,const float tan_fovx, const float tan_fovy,
						const int iter,
						const int num_spp,
						float* db0,
						float* db1,
						float* dV)
{	
	
	int idx=cg::this_grid().thread_rank();
	if(idx>=H*W)
	{
		return;
	}

	int tri_id=intersected_tris[num_spp*idx+iter];
	if(tri_id<0)
	{
		db0[idx]=0.0f;
		db1[idx]=0.0f;
		return;
	}

	int bias_idx=2*H*W*iter+2*idx;
	vec2f bias=vec2f(ray_bias[bias_idx],ray_bias[bias_idx+1]);
	Ray ray=generateRay(idx, iter,cam_to_world, bias,tan_fovx, tan_fovy,  H,  W);


	//只需要顶点信息
	Primitive p;
	int ind=triangles[3*tri_id];
	p.points[0]=vec3f(vertices[3*ind],vertices[3*ind+1],vertices[3*ind+2]);
	ind=triangles[3*tri_id+1];
	p.points[1]=vec3f(vertices[3*ind],vertices[3*ind+1],vertices[3*ind+2]);
	ind=triangles[3*tri_id+2];
	p.points[2]=vec3f(vertices[3*ind],vertices[3*ind+1],vertices[3*ind+2]);



	float time=0.0f;
	float b0=0.0f;
	float b1=0.0f;
	TriangleRayInsect(p, ray, time, b0,b1);

	float b2=1.0f-b0-b1;
	vec3f dLdp0=vec3f(0,0,0);
	vec3f dLdp1=vec3f(0,0,0);
	vec3f dLdp2=vec3f(0,0,0);
	float dLdb0=db0[idx];
	float dLdb1=db1[idx];

	TriangleGradCompute(p, ray,dLdb0,dLdb1, dLdp0, dLdp1, dLdp2);
	
	//给顶点计算梯度
	//先除以spp

	float inv_spp=1.0f/num_spp;
	dLdp0*=inv_spp;
	dLdp1*=inv_spp;
	dLdp2*=inv_spp;

	ind=triangles[3*tri_id];
	atomicAdd(&(dV[3*ind]),dLdp0.x );
	atomicAdd(&(dV[3*ind+1]),dLdp0.y );
	atomicAdd(&(dV[3*ind+2]),dLdp0.z );

	ind=triangles[3*tri_id+1];
	atomicAdd(&(dV[3*ind]),dLdp1.x );
	atomicAdd(&(dV[3*ind+1]),dLdp1.y );
	atomicAdd(&(dV[3*ind+2]),dLdp1.z );

	ind=triangles[3*tri_id+2];
	atomicAdd(&(dV[3*ind]),dLdp2.x );
	atomicAdd(&(dV[3*ind+1]),dLdp2.y );
	atomicAdd(&(dV[3*ind+2]),dLdp2.z );

	
	//不要忘记给dL db0 dL db1置0进行下一次循环
	db0[idx]=0.0f;
	db1[idx]=0.0f;

}


void PathTraceBWD(
					int* intersected_tris,
					int* texture_id,
					const int* triangles,
					const float* vertices,
					const float* uvs,
					const int* uv_index,
					const float* texture,
					const float* cam_to_world,
					const float* ray_bias,
					const int H,
					const int W,
					const int texture_H,
					const int texture_W,
					const float tan_fovy,
					const float tan_fovx,
					const int num_spp,
					const float* dO,
					float* d_b0,
					float* d_b1,
					float* d_vertices,
					float* d_uv,
					float* d_texture)
{
	int j;
	int num_rays=H*W;

	cudaError_t err;

	for(j=0;j<num_spp;j++)
	{
		ComputeUV<<<(num_rays+255)/256,256>>>(intersected_tris,triangles,vertices,uvs,uv_index,texture,texture_id,ray_bias,cam_to_world, 
												H, W,texture_H,texture_W,tan_fovx,tan_fovy,j,num_spp,dO,d_b0,d_b1,d_texture,d_uv);
		cudaDeviceSynchronize();
		err=cudaGetLastError();
		if(err!=cudaSuccess)
		{
			printf("计算uv梯度失败\n");
		}

		ComputedV<<<(num_rays+255)/256,256>>>(intersected_tris,triangles,vertices,ray_bias,cam_to_world,
						H, W,tan_fovx,tan_fovy,j,num_spp,d_b0,d_b1,d_vertices);
		cudaDeviceSynchronize();
		err=cudaGetLastError();
		if(err!=cudaSuccess)
		{
			printf("计算顶点梯度失败\n");
		}

	}
	
}
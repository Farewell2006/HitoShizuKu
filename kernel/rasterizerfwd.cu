#include <rasterizer.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda.h>
#include <fstream>
#include <cub/cub.cuh>

namespace cg=cooperative_groups;

//为辅助数据分配内存并对齐
GeometryInfo GeometryInfo::ALLOC(char*& address, size_t N)
{
	GeometryInfo g;
	manual_allocate(address, g.depth, N, 128);
	manual_allocate(address, g.screen_coor, N, 128);

	return g;
}

IntermediateInfo IntermediateInfo::ALLOC(char*& address, size_t N)
{
	IntermediateInfo i;
	manual_allocate(address, i.tiles_intersected, N, 128);
	manual_allocate(address, i.scanned_res, N, 128);

	cub::DeviceScan::InclusiveSum(nullptr,i.scan_size,i.tiles_intersected,i.tiles_intersected,N);
	manual_allocate(address, i.scan_address, i.scan_size, 128);

	return i;

}

TriangleInfo TriangleInfo::ALLOC(char*& address, size_t N)
{
	TriangleInfo T;
	manual_allocate(address, T.keysUnsorted, N, 128);
	manual_allocate(address, T.keysSorted, N, 128);
	manual_allocate(address, T.TriangleIndexUnsorted, N, 128);
	manual_allocate(address, T.TriangleIndexSorted, N, 128);
	cub::DeviceRadixSort::SortPairs(nullptr, T.sort_size, T.keysUnsorted,T.keysSorted, T.TriangleIndexUnsorted,T.TriangleIndexSorted,N);

	manual_allocate(address, T.sortAddress, T.sort_size, 128);

	return T;
}

ImageInfo ImageInfo::ALLOC(char*& address, size_t N)
{
	ImageInfo img;
	manual_allocate(address, img.ranges, N, 128);

	return img;
}

__global__ void ProjectVertices(int num_vertices, const float* vertices, const float* proj_matrix, float* depth,float2* screen_coor,int H,int W)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_vertices)
	{
		return;
	}
	float3 point={vertices[3*idx],vertices[3*idx+1],vertices[3*idx+2]};
	float3 res=ScreenProject(proj_matrix, point);

	screen_coor[idx].x=W-res.x;
	screen_coor[idx].y=H-res.y;
	depth[idx]=res.z;

}

__global__ void PrepTriangles(int num_triangles, const int* triangles,float2* screen_coor, uint32_t* tiles_insected,dim3 grid)
{
	int idx=cg::this_grid().thread_rank();

	if(idx>=num_triangles)
	{
		return;
	}
	tiles_insected[idx]=0;
	//获得三角形的三个顶点在屏幕上的坐标
	float2 v1=screen_coor[triangles[3*idx]];
	float2 v2=screen_coor[triangles[3*idx+1]];
	float2 v3=screen_coor[triangles[3*idx+2]];


	//计算这个三角形大概和多少个瓦片相交

	uint2 left_bottom;
	uint2 right_up;

	AABB(v1,v2, v3, left_bottom,right_up, grid);
	tiles_insected[idx]=(right_up.x-left_bottom.x)*(right_up.y-left_bottom.y);
}

__global__ void GenerateKeyList(int num_triangles,const int* triangles, float2* screen_coor, 
								uint32_t* tiles_intersected,uint32_t* scanned_res, 
								uint32_t* keysUnsorted ,uint32_t* TriangleIndexUnsorted, dim3 grid)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_triangles)
	{
		return;
	}

	uint32_t start=(idx==0)?0:scanned_res[idx-1];
	uint2 left_bottom, right_up;

	float2 v1=screen_coor[triangles[3*idx]];
	float2 v2=screen_coor[triangles[3*idx+1]];
	float2 v3=screen_coor[triangles[3*idx+2]];

	AABB(v1,v2, v3, left_bottom,right_up, grid);

	int i,j;
	for(i=left_bottom.x ;i<right_up.x;i++)
	{
		for(j=left_bottom.y;j<right_up.y;j++)
		{
			keysUnsorted[start]=j*grid.x+i;
			TriangleIndexUnsorted[start]=idx;
			start+=1;
		}
	}
}

__global__ void SetZero(int H,int W, uint2* ranges)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=H*W)
	{
		return;
	}
	ranges[idx].x=0;
	ranges[idx].y=0;
}

__global__ void GetTileRanges(int num_rastered, uint2* ranges, uint32_t* keysSorted)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_rastered)
	{
		return;
	}
	
	uint32_t current_key=keysSorted[idx];
	if(idx==0)
	{
		ranges[current_key].x=0;
	}
	else
	{
		
		uint32_t pre_key=keysSorted[idx-1];
		if(current_key>pre_key)
		{
			ranges[current_key].x=idx;
			ranges[pre_key].y=idx;
		}
	}
	if(idx==num_rastered-1)
	{
		ranges[current_key].y=num_rastered;
	}
}

__device__ bool BilinearSample( float2 uv, int texture_H, int texture_W,int& start_x, int& start_y)
{
	int u=(int)(uv.x);
	int v=(int)(uv.y);

	u=max(0,u);
	v=max(0,v);

	u=min(texture_W-1,u);
	v=min(texture_H-1,v);

	if( u>=texture_W || v>=texture_H )
	{
		return false;
	}

	start_x=u-1;
	start_y=v-1;
	if(u%2==0)
	{
		start_x=u;
	}
	if(v%2==0)
	{
		start_y=v;
	}
	return true;	
}


__global__ void Shade( const float* background, const int* triangles, uint32_t* TriangleIndexSorted ,const float*  texture, const int* f_mtl ,
						float2* screen_coor , float* depth,const float2* uvs ,const int* uv_index,
						uint2* ranges,int H,int W, int texture_H,int texture_W,
						float* res)
{
	//获取该线程所在的block
	cg::thread_block block=cg::this_thread_block();
	int xblock_num=(W+BLOCK_X-1)/BLOCK_X;

	//每个block和一个tile唯一对应, 现在获得这个tile的工作范围
	uint2 range= ranges[ block.group_index().y*xblock_num + block.group_index().x];
	int remain=range.y-range.x;
	int tail=remain;
	
	uint2 pix={block.group_index().x*BLOCK_X+block.thread_index().x, block.group_index().y*BLOCK_Y+block.thread_index().y};
	float2 pixf={(float)pix.x+0.5f,(float)pix.y+0.5f};
	int pix_idx=pix.y*W+pix.x;

	bool valid=true;
	if( pix.x>=W ||pix.y>=H )
	{
		valid=false;
	}

	
	if(valid)
	{
		res[3*pix_idx]=0.0f;
		res[3*pix_idx+1]=0.0f;
		res[3*pix_idx+2]=0.0f;
	}

	//共享内存
	__shared__ float2 coor[BLOCK_SIZE][3];
	__shared__ float depthBuffer[BLOCK_SIZE][3];
	__shared__ float2 uv[BLOCK_SIZE][3];
	__shared__ uint32_t tri_index[BLOCK_SIZE];


	int i=0;
	int j=0;
	int k=0;
	float dep[4]={10000.0f,10000.0f,10000.0f,10000.0f};
	float3 colorBuffer[4];
	colorBuffer[0]={background[0],background[1],background[2]};
	colorBuffer[1]={background[0],background[1],background[2]};
	colorBuffer[2]={background[0],background[1],background[2]};
	colorBuffer[3]={background[0],background[1],background[2]};
	
	for(i=0;i< (remain+BLOCK_SIZE-1)/BLOCK_SIZE && tail>0;tail-=BLOCK_SIZE,i+=1)
	{
		//把数据读入共享内存
		int current_id=i*BLOCK_SIZE+block.thread_rank();
		if(current_id<remain)
		{
			uint32_t tri_id=TriangleIndexSorted[range.x+current_id];
			tri_index[block.thread_rank()]=tri_id;
			int id1,id2,id3;
			id1=triangles[3*tri_id];
			id2=triangles[3*tri_id+1];
			id3=triangles[3*tri_id+2];
			
			coor[block.thread_rank()][0]=screen_coor[id1 ];
			coor[block.thread_rank()][1]=screen_coor[ id2];
			coor[block.thread_rank()][2]=screen_coor[ id3];

			depthBuffer[block.thread_rank()][0]=depth[id1];
			depthBuffer[block.thread_rank()][1]=depth[id2];
			depthBuffer[block.thread_rank()][2]=depth[id3];

			uv[block.thread_rank()][0]=   uvs[ uv_index[3*tri_id] ] ;
			uv[block.thread_rank()][1]=uvs[ uv_index[3*tri_id+1] ];
			uv[block.thread_rank()][2]=uvs[ uv_index[3*tri_id+2] ];
		}

		block.sync();
		//开始成像
		for(j=0;valid && j<min(tail,BLOCK_SIZE);j++)
		{
			float b1=0.0f,b2=0.0f;
			//相交了就往里写东西
			for(k=0;k<4;k++)
			{
				int2 bias={k/2,k%2};
				float2 sampled_point={bias.x*0.25+pixf.x-0.25,bias.y*0.25+pixf.y-0.25 };
				if( CheckTriangle(coor[j][0], coor[j][1], coor[j][2],sampled_point,  b1,  b2) )
				{
					float alpha=b1/depthBuffer[j][0];
					float beta=b2/depthBuffer[j][1];
					float gamma=(1.0f-b1-b2)/depthBuffer[j][2];
					//首先是深度检测
					float tar_dep= 1.0f/(alpha+beta+gamma);
					//如果更小的话就可以替换当前的像素值
					if(tar_dep<dep[k] && tar_dep>0)
					{
						//透视投影采样uv
					
						float2 pix_uv=  {alpha*uv[j][0].x+beta*uv[j][1].x+gamma*uv[j][2].x,alpha*uv[j][0].y+beta*uv[j][1].y+gamma*uv[j][2].y };
						pix_uv.x/=alpha+beta+gamma;
						pix_uv.y/=alpha+beta+gamma;

						//双线性采样会顺便检查uv坐标是否合法
						pix_uv.x*=texture_W;
						pix_uv.y*=texture_H;
						int start_x=0,start_y=0;
						
						if(BilinearSample( pix_uv, texture_H, texture_W,start_x,start_y))
						{
							uint32_t text_bias=f_mtl[tri_index[j]]*(texture_H*texture_W*3);

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

							colorBuffer[k].x=(start_y+1-pix_uv.y)*x0+(pix_uv.y-start_y)*x1;
							colorBuffer[k].y=(start_y+1-pix_uv.y)*y0+(pix_uv.y-start_y)*y1;
							colorBuffer[k].z=(start_y+1-pix_uv.y)*z0+(pix_uv.y-start_y)*z1;

							dep[k]=tar_dep;
						}					
					}
				}
			}
		}
		block.sync();
	}
	
	for(k=0;k<4;k++)
	{
		res[3*pix_idx]+=0.25*colorBuffer[k].x;
		res[3*pix_idx+1]+=0.25*colorBuffer[k].y;
		res[3*pix_idx+2]+=0.25*colorBuffer[k].z;
	}

}

void RasterizationFWD( 
						std::function<char*(size_t)> geometryInfo,
						std::function<char*(size_t)> intermediateInfo,
						std::function<char*(size_t)> triangleInfo,
						std::function<char* (size_t)> imageInfo,
						int num_vertices,
						int num_triangles,
						int num_uvs,
						const float* background,
						const int* triangles,
						const float* vertices,
						const float* uvs,
						const int* uv_index,
						const float* texture,
						const int* f_mtl,
						const float* proj_matrix,
						const float tan_fovx,
						const float tan_fovy,
						const int H,
						const int W,
						const int texture_H,
						const int texture_W,
						float* res)
{

	size_t chunk_size=PrecomputeBias<GeometryInfo>(num_vertices);
	char* chunk_ptr=geometryInfo(chunk_size);
	GeometryInfo g=GeometryInfo::ALLOC(chunk_ptr, num_vertices);


	dim3 block_dim(BLOCK_X,BLOCK_Y,1);


	// step1. 把顶点的屏幕坐标和深度值写入geometryInfo g
	ProjectVertices<<<(num_vertices+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(num_vertices, vertices, proj_matrix, g.depth,g.screen_coor,H,W);
	//end of step 1


	size_t intermediate_chunk_size=PrecomputeBias<IntermediateInfo>(num_triangles);
	char* intermediate_ptr=intermediateInfo(intermediate_chunk_size);
	IntermediateInfo i=IntermediateInfo::ALLOC(intermediate_ptr,num_triangles);

	dim3 grid_dim((W+BLOCK_X-1)/BLOCK_X, (H+BLOCK_Y-1)/BLOCK_Y);



	

	//step2. 预处理三角形,计算每个三角形大概和多少个tile相交并把结果放入intermediateInfo i
	PrepTriangles<<<(num_triangles+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(num_triangles,triangles,g.screen_coor,i.tiles_intersected,grid_dim);
	//end of step2

	//step3 调用前缀和看看有多少三角形需要光栅化
	cub::DeviceScan::InclusiveSum(i.scan_address,i.scan_size,i.tiles_intersected,i.scanned_res,num_triangles);
	//end of step3


	int raster_size=0;

	//总数赋值给 raster_size
	cudaMemcpy(&raster_size, i.scanned_res+num_triangles-1 ,sizeof(int),cudaMemcpyDeviceToHost);

	
	
	size_t triangle_chunk_size=PrecomputeBias<TriangleInfo>(raster_size);
	char* triangle_ptr=triangleInfo(triangle_chunk_size);
	TriangleInfo t=TriangleInfo::ALLOC(triangle_ptr,raster_size);



	//step4 把每个三角形和它可能相交的tile关联起来，然后按照tile排序
	GenerateKeyList<<< (num_triangles+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE >>>(num_triangles,triangles, g.screen_coor, 
								i.tiles_intersected,i.scanned_res, 
								t.keysUnsorted ,t.TriangleIndexUnsorted, grid_dim);
	//end of step4



	//step5 给三角形索引按照tile的序号排序 
	cub::DeviceRadixSort::SortPairs(t.sortAddress, t.sort_size, t.keysUnsorted,t.keysSorted, t.TriangleIndexUnsorted,t.TriangleIndexSorted,raster_size,0,32 );
	//end of step5



	size_t img_chunk_size=PrecomputeBias<ImageInfo>(H*W);
	char* img_chunk_ptr=imageInfo(img_chunk_size);
	ImageInfo img=ImageInfo::ALLOC(img_chunk_ptr,H*W);


	//step6 从排序结果中获得可能和每个tile相交的三角形的索引 在这之前先给img.ranges初始化 避免内存访问错误
	SetZero<<< grid_dim,block_dim>>>(H,W, img.ranges);



	GetTileRanges<<< (raster_size+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE >>>(raster_size,img.ranges, t.keysSorted);
	//end of step6


	//step 7 着色
	Shade<<<grid_dim,block_dim>>>(background,triangles, t.TriangleIndexSorted ,texture,f_mtl,g.screen_coor , g.depth,(const float2*)uvs,uv_index,img.ranges,H,W,texture_H,texture_W,  res);
	//end of step7


} 
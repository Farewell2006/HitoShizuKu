#include <3dgs_forward.h>
#include <3dgs_structure.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <fstream>
#include <cuda.h>

namespace cg=cooperative_groups;

//强制内联的好处是优化函数调用时的开销
__forceinline__ __device__ bool in_frustum(const float3 position,const float* proj_matrix,const float* view_matrix, float3 & p_ndc )
{
	
	//view_matrix和proj_matrix都是row major存储的, 先计算点在相机坐标系下的坐标 (不需要最后一个分量是因为齐次坐标最后一个肯定是1)
	float4 screen_coor={	view_matrix[0]*position.x+view_matrix[1]*position.y+view_matrix[2]*position.z+view_matrix[3],
						view_matrix[4]*position.x+view_matrix[5]*position.y+view_matrix[6]*position.z+view_matrix[7],
						view_matrix[8]*position.x+view_matrix[9]*position.y+view_matrix[10]*position.z+view_matrix[11],
						view_matrix[12]*position.x+view_matrix[13]*position.y+view_matrix[14]*position.z+view_matrix[15]};
	
	float dep=proj_matrix[8]*position.x+proj_matrix[9]*position.y+proj_matrix[10]*position.z+proj_matrix[11];
	
	//float inv_w=1.0/(ndc_coor.w+EPS);
	float inv=1.0f/screen_coor.w;
	p_ndc={screen_coor.x*inv,screen_coor.y*inv,dep};
	if(p_ndc.z<0.0f)
	{
		return false;
	}
	
	return true;
} 

__device__ void computeCovariance3D(const float3 scale, float scale_modifier, const float4 quant_num,float* cov3D)
{
	//按照论文中的公式6计算协方差矩阵\Sigma scale相当于S的对角线, 不要忘记给S乘以scale_modifier
	//quant_num代表3维旋转的四元数, 用论文的附录中的公式(10)变成3阶旋转矩阵R
	//由于协方差矩阵是对角阵, 所以只需存储右上角的6个元素

	float R[9];
	float r=quant_num.x;
	float i=quant_num.y;
	float j=quant_num.z;
	float k=quant_num.w;
	// row major
	R[0]=2*(0.5-j*j-k*k);
	R[1]=2*(i*j-r*k);
	R[2]=2*(i*k+r*j);
	R[3]=2*(i*j+r*k);
	R[4]=2*(0.5-i*i-k*k);
	R[5]=2*(j*k-r*i);
	R[6]=2*(i*k-r*j);
	R[7]=2*(j*k-r*i);
	R[8]=2*(0.5-i*i-j*j);

	//M=RS
	float M[9];
	M[0]=scale_modifier*scale.x*R[0];
	M[1]=scale_modifier*scale.y*R[1];
	M[2]=scale_modifier*scale.z*R[2];
	M[3]=scale_modifier*scale.x*R[3];
	M[4]=scale_modifier*scale.y*R[4];
	M[5]=scale_modifier*scale.z*R[5];
	M[6]=scale_modifier*scale.x*R[6];
	M[7]=scale_modifier*scale.y*R[7];
	M[8]=scale_modifier*scale.z*R[8];

	//Sigma=M*MT, 只存右上角, row_major
	cov3D[0]=M[0]*M[0]+M[1]*M[1]+M[2]*M[2];
	cov3D[1]=M[0]*M[3]+M[1]*M[4]+M[2]*M[5];
	cov3D[2]=M[0]*M[6]+M[1]*M[7]+M[2]*M[8];
	cov3D[3]=M[3]*M[3]+M[4]*M[4]+M[5]*M[5];
	cov3D[4]=M[3]*M[6]+M[4]*M[7]+M[5]*M[8];
	cov3D[5]=M[6]*M[6]+M[7]*M[7]+M[8]*M[8];

	
}
__device__ float3 computeCovariance2D(const float3& position,const float* view_matrix,
								const float focal_x, 
								const float focal_y, 
								const float tan_fovx, 
								const float tan_fovy,
								float* cov3D)
{
	//把position变成相机空间的坐标
	float3 cam_coor={	view_matrix[0]*position.x+view_matrix[1]*position.y+view_matrix[2]*position.z+view_matrix[3],
						view_matrix[4]*position.x+view_matrix[5]*position.y+view_matrix[6]*position.z+view_matrix[7],
						view_matrix[8]*position.x+view_matrix[9]*position.y+view_matrix[10]*position.z+view_matrix[11]};
	float tx=cam_coor.x;
	float ty=cam_coor.y;
	float tz=cam_coor.z;
	

	const float limx=1.3f*tan_fovx;
	const float limy=1.3f*tan_fovy;
	float xdevidez=tx/tz;
	float ydevidez=ty/tz;
	tx=(float)min(limx,max(xdevidez,-limx))*tz;
	ty=(float)min(limy,max(ydevidez,-limy))*tz;
	


	//实际上J的最后一行不是0，但我们最终的计算结果只取三维矩阵的左上二阶分块, 这个二阶分块是不受J的第三行影响的.
	mat3 J={
			focal_x/tz,0,-1.0*focal_x*tx/(tz*tz),
			0,focal_y/tz,-1.0*focal_y*ty/(tz*tz),
			0,0,0};
	

	//W是view矩阵的旋转部分, 即左上的三阶分块
	mat3 W={
			view_matrix[0],view_matrix[1],view_matrix[2],
			view_matrix[4],view_matrix[5],view_matrix[6],
			view_matrix[8],view_matrix[9],view_matrix[10]};
	//构建协方差矩阵, 注意cov3D里面存的是右上角的元素
	mat3 sigma={
				cov3D[0],cov3D[1],cov3D[2],
				cov3D[1],cov3D[3],cov3D[4],
				cov3D[2],cov3D[4],cov3D[5]};

	
	//按照论文的公式(5)完成计算.
	mat3 new_sigma=matmul(sigma,transpose(W));
	new_sigma=matmul(W,new_sigma);
	new_sigma=matmul(new_sigma,transpose(J));
	new_sigma=matmul(J,new_sigma);

	//同样的 左上分块肯定是对角阵, 我们只存对角线和右上角的元素 总共三个
	//给对角线元素+0.3f是为了保证每个gaussian能够占据一个像素.
	
	return {new_sigma.data[0]+0.3f,new_sigma.data[1],new_sigma.data[4]+0.3f};
	

}

__global__ void GenerateKeyList(int num_points,float2* means2D, float* depth, uint32_t* scanned_points,uint64_t* keys_unsorted, uint32_t* point_unsorted, int* radius, dim3 grid)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_points)
	{
		return;
	}
	int i,j;
	//半径为0的Gaussian对图片没有任何贡献，直接无视
	if(radius[idx]>0)
	{
		uint32_t offset=(idx==0)? 0:scanned_points[idx-1];
		uint2 rect_min, rect_max;
		getRect(means2D[idx],radius[idx],rect_min,rect_max,grid);
		
		for(i=rect_min.x;i<rect_max.x;i++)
		{
			for(j=rect_min.y;j<rect_max.y;j++)
			{
				uint64_t key=j*grid.x+i;
				key<<=32;
				key|=*((uint32_t*)&depth[idx]);
				keys_unsorted[offset]=key;
				point_unsorted[offset]=idx;
				offset+=1;

			}
		}
	}
}

__global__ void PreprocessKernel(int num_points, int D, int M,
								const float* means3D, 
								const float* ori_colors,
								const float* scale,
								const float scale_modifier, 
								const float* quant_number,
								const float* opacity, 
								const float* spherical_harmoincs, 
								const float* view_matrix, 
								const float* proj_matrix, 
								const float* camera_position,
								const int H,
								const int W, 
								const float focal_x, 
								const float focal_y, 
								const float tan_fovx, 
								const float tan_fovy, 
								int* radius, 
								float2* means2D, 
								float* depth,
								float* cov3D, 
								float* color, 
								float4* conic_opacity, 
								dim3 grid, 
								uint32_t* insected_tiles,
								float* position2d,
								float* cov2d)             //这个预计算的Gaussian点讲落在哪个tile里面
								
{
	//尝试用cg::this_grid().thread_rank()获得这个线程的序号, 请自行询问AI或查阅文档! 不要用传统方法 idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idx=cg::this_grid().thread_rank();
	
	//并行预处理的思路是每个thread处理一个Guassian点
	if(idx>=num_points)
	{
		return;
	}
	
	//初始化我们需要求解的数据
	radius[idx]=0;
	insected_tiles[idx]=0;
	
	//有些Gaussian点是不在视椎体当中的, 这些点不用考虑, 现在实现一个判断Gaussian点是否在视椎体中的函数: in_frustum 
	float3 position={means3D[idx*3], means3D[idx*3+1],means3D[idx*3+2]};
	// p_ndc就是设备空间的坐标, 前两个为x,y 最后一个为深度, 必定比0大（小于0的不会被相机看见)
	float3 p_ndc;
	if(!in_frustum(position,view_matrix,proj_matrix, p_ndc))
	{
		return;
	}
	
	//计算这个3DGaussian的协方差矩阵
	float3 p_scale={scale[3*idx],scale[3*idx+1],scale[3*idx+2]};
	float4 quant_num={quant_number[4*idx],quant_number[4*idx+1],quant_number[4*idx+2],quant_number[4*idx+3]};
	//对于每个点, 因为协方差矩阵是对称的, 所以只需存储6个元素 所以是+6*idx
	computeCovariance3D(p_scale, scale_modifier, quant_num, cov3D+6*idx); 

	//计算屏幕空间中的Gaussian点
	float3 cov2D=computeCovariance2D(position,view_matrix,focal_x, focal_y, tan_fovx,tan_fovy,cov3D+6*idx);

	//用伴随矩阵的方式计算2维协方差矩阵的逆矩阵 如果不可逆就放弃这个矩阵
	float det=cov2D.x*cov2D.z-cov2D.y*cov2D.y;
	if(det==0.0f)
	{
		return;
	}
	float inv_det=1.0f/det;
	float3 conv2D_inv={cov2D.z*inv_det, -cov2D.y*inv_det,cov2D.x*inv_det};

	//p_ndc已经在in_frustum中被赋值为ndc坐标, x,y为[-1,1]中的数,z是深度
	float2 screen_coordinate={W-p_ndc.x,H- p_ndc.y};

	//计算2D Gaussian球体的半径(即协方差矩阵的特征值的平方和)
	float b=0.5*(cov2D.x+cov2D.z);
	float delta=sqrt(max(0.1f,b*b-det));
	float lambda1=b+delta;
	float lambda2=b-delta;
	int max_radius=(int)ceil(3.0f*sqrt(max(lambda1,lambda2)));
	uint2 left_bottom, right_up;

	getRect(screen_coordinate, max_radius, left_bottom, right_up, grid);
	if((right_up.x-left_bottom.x)*(right_up.y-left_bottom.y)==0)
	{
		return;
	}

	depth[idx]=p_ndc.z;
	radius[idx]=max_radius;
	means2D[idx]=screen_coordinate;
	

	position2d[3*idx]=means2D[idx].x;
	position2d[3*idx+1]=means2D[idx].y;
	position2d[3*idx+2]=p_ndc.z;

	cov2d[3*idx]=conv2D_inv.x;
	cov2d[3*idx+1]=conv2D_inv.y;
	cov2d[3*idx+2]=conv2D_inv.z;

	conic_opacity[idx]=float4{conv2D_inv.x,conv2D_inv.y,conv2D_inv.z,opacity[idx] };
	insected_tiles[idx]=(right_up.x-left_bottom.x)*(right_up.y-left_bottom.y);
	for(int k=0;k<3;k++)
	{
		color[idx*3+k]=ori_colors[idx*3+k];
	}
	
}

__global__ void GetBlockRange(int num_rendered, uint64_t* keys,uint2* range)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_rendered)
	{
		return;
	}
	//从key中获取gird的序号，考虑位运算
	uint32_t currentblock=keys[idx]>>32;
	if(idx==0)
	{
		range[currentblock].x=0;
	}
	else
	{
		//上一个block的序号
		uint32_t preblock=keys[idx-1]>>32;
		
		//不相等的话说明是分界点
		if(currentblock!=preblock)
		{
			range[preblock].y=idx;
			range[currentblock].x=idx;
		}
	}
	//最后一个block, 这个block的终止节点还没定义
	if(idx==num_rendered-1)
	{
		range[currentblock].y=num_rendered;
	}

}

__global__ void splate(
						const float* background, uint2* ranges,uint32_t* sorted_point_idx, int H,int W, 
						float2* means2D,float* colors,float4* invcoc2D_opacity,
						float* alpha, uint32_t* num_contributor, float* res)
{
	//获取当前线程所在的block
	cg::thread_block block=cg::this_thread_block();
	uint32_t x_blocks=(W+BLOCK_X-1)/BLOCK_X;
	//这个block在图像中负责哪些像素点
	uint2 pix_min={block.group_index().x*BLOCK_X,block.group_index().y*BLOCK_Y};
	uint2 pix_max={min(W,block.group_index().x*BLOCK_X+BLOCK_X),min(H,block.group_index().y*BLOCK_Y+BLOCK_Y)};

	uint2 pix={pix_min.x+block.thread_index().x, pix_min.y+ block.thread_index().y};
	uint32_t pix_id=pix.y*W+pix.x;

	float2 pixf={(float)pix.x,(float)pix.y};

	bool done=!(pix.x<W && pix.y<H);

	//读取这个block应该处理的Gaussian的序号
	uint2 range=ranges[block.group_index().y*x_blocks+block.group_index().x];
	
	//一个thread至多跑多少轮
	const int rounds=((range.y-range.x+BLOCK_SIZE-1)/BLOCK_SIZE);
	int toDo=range.y-range.x;

	//共享内存
	__shared__ int index[BLOCK_SIZE];
	__shared__ float2 coordinates[BLOCK_SIZE];
	__shared__ float4 cov2dinv_and_opacity[BLOCK_SIZE];

	//按照论文的公式(2)和(3)完成渲染
	float T=1.0;
	uint32_t contributor=0;
	uint32_t last_contributor=0;
	float C[3]={0};

	int i=0;
	int j=0;
	for(i=0;i<rounds;i++,toDo-=BLOCK_SIZE)
	{
		int num_done=__syncthreads_count(done);
		if(num_done==BLOCK_SIZE)
		{
			break;
		}

		int current=i*BLOCK_SIZE+block.thread_rank();
		if(range.x+current<range.y)
		{
			//找到需要处理的Gaussian的id
			
			int pid=sorted_point_idx[range.x+current];
			
			index[block.thread_rank()]=pid;
			coordinates[block.thread_rank()]=means2D[pid];
			cov2dinv_and_opacity[block.thread_rank()]=invcoc2D_opacity[pid];
			
		}
		block.sync();

		for(j=0;!done && j<min(BLOCK_SIZE,toDo);j++)
		{
			contributor+=1;
			float2 xy=coordinates[j];
			float2 dist={xy.x-pixf.x,xy.y-pixf.y};
			float4 cov=cov2dinv_and_opacity[j];
			
			//计算Gaussian的指数
			float exponent=-0.5*(cov.x*dist.x*dist.x+cov.z*dist.y*dist.y)-cov.y*dist.x*dist.y;
			if(exponent>0.0f)
			{
				continue;
			}

			float alphas=min(0.99f,cov.w*exp(exponent));
			if(alphas<1.0f/255.0f)
			{
				continue;
			}
			float current_T =T*(1.0f-alphas);
			if(current_T<0.0001f)
			{
				done=true;
				continue;
			}

			for(int k=0;k<3;k++)
			{
				
				C[k]+=alphas*current_T*colors[index[j]*3+k];
			}
			T=current_T;
			last_contributor=contributor;

		}
	}

	if(pix.x<W && pix.y<H)
	{
		alpha[pix_id]=T;

		num_contributor[pix_id]=last_contributor;
		for(int k=0;k<3;k++)
		{
			res[k*H*W+pix_id]=C[k]+T*background[k];
		}
	}

}

uint32_t FindMSB(uint32_t n)
{
	//找出n的二进制展开中左边第一个不是0的数的位数，设最右侧是第0位，往左位数增加
	uint32_t msb=sizeof(n)*4;
	uint32_t step=msb;
	while(step>1)
	{
		step/=2;
		if(n>>msb)
		{
			msb+=step;
		}
		else
		{
			msb-=step;
		}
	}
	if(n>>msb)
	{
		msb+=1;
	}
	return msb;
}

int gsForward(  std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binnBuffer,
	std::function<char* (size_t)> ImgBuffer,
	const float* background,
	int num_points,
	int D, int M,
	const float* means3D,
	const float* colors,
	const float* scale,
	const float scale_modifier,
	const float* quant_number,
	const float* opacity,
	const float* spherical_harmoincs,
	const float* view_matrix,
	const float* project_matrix,
	const float* camera_position,
	const int H,
	const int W,
	const float tan_fovx,
	const float tan_fovy,
	int* radius,
	float* res,
	float* position2d,
	float* cov2d)
{
	const float focal_x=(float)W/(2.0*tan_fovx);
	const float focal_y=(float)H/(2.0*tan_fovy);


	size_t chunk_size=required<Geometry>(num_points);


	

	//回忆std::function<char*(size_t)>返回一个地址
	char* chunk_ptr=geometryBuffer(chunk_size);
	Geometry g=Geometry::fromChunk(chunk_ptr,num_points);

	dim3 grid_dim((W+BLOCK_X-1)/BLOCK_X,(H+BLOCK_Y-1)/BLOCK_Y);
	dim3 block_dim(BLOCK_X,BLOCK_Y,1);

	PreprocessKernel<<<(num_points+255)/256,256>>>( num_points, D,  M, means3D, colors,
								 scale,
								 scale_modifier, 
								 quant_number,
								 opacity, 
								 spherical_harmoincs, 
								view_matrix, 
								project_matrix, 
								camera_position,
								H,
								W, 
								focal_x, 
								focal_y, 
								tan_fovx, 
								tan_fovy, 
								g.radius, 
								g.means2D, 
								g.depth,
								g.cov3D, 
								g.colors, 
								g.invcoc2D_opacity, 
								grid_dim, 
								g.insected_tiles,
								position2d,
								cov2d);

	cudaError_t err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("Kernel launch failed: %s\n",cudaGetErrorString(err));
	}
	//做一次扫描获得前缀和 g.scanned_points[idx]-g.scanned_points[idx-1]就是第idx个点要着色的次数 
	cub::DeviceScan::InclusiveSum(g.scan_address,g.scan_size,g.insected_tiles,g.scanned_points,num_points);
	int num_rendered;
	cudaMemcpy(&num_rendered,g.scanned_points+num_points-1,sizeof(int),cudaMemcpyDeviceToHost );

	size_t binn_chunk_size=required<Binning>(num_rendered);
	char* binn_chunk_ptr=binnBuffer(binn_chunk_size);


	Binning b=Binning::fromChunk(binn_chunk_ptr,num_rendered);
	GenerateKeyList<<<(num_points+255)/256,256>>>(num_points,g.means2D, g.depth, g.scanned_points,b.keys_unsorted,b.pointsIdx_unsorted,g.radius, grid_dim);
	if(err!=cudaSuccess)
	{
		printf("Kernel launch failed: %s\n",cudaGetErrorString(err));
	}
	//鉴于图片大小的限制, 尽管key是64位的整数，但是我们所有的key未必能占满全部的左边32位，所以可以考虑把左边都一样的地方去掉，提升排序效率
	//具体的说，一个key是由 [grid_Idx| depth]组成的 左边32位是grid_idx，然而所有grid得数目未必有2^{33}-1这么多，所以只需计算最大的grid_idx是多少，然后计算出这个数
	//左边不是1的位数bit，排序时只需比较key右侧32+bit的位数即可

	int bit =FindMSB(grid_dim.x*grid_dim.y);
	cub::DeviceRadixSort::SortPairs(b.list_sorting_space,b.sort_size,b.keys_unsorted,b.keys_sorted,b.pointsIdx_unsorted,b.pointIdx_sorted,num_rendered,0, 32+bit);


	

	size_t img_size=required<Image>(H*W);
	char* img_chunk_ptr=ImgBuffer(img_size);
	Image img=Image::fromChunk(img_chunk_ptr,H*W);

	

	//确定每个block的工作范围, 对每个key，第33到32+bit位就是这个gaussian点所在的block的位置
	GetBlockRange<<<(num_rendered+255)/256,256>>>(num_rendered, b.keys_sorted,img.ranges);




	if(err!=cudaSuccess)
	{
		printf("Kernel launch failed: %s\n",cudaGetErrorString(err));
	}

	

	//每个block处理它负责的Gaussian点
	splate<<<grid_dim,block_dim>>>(background, img.ranges,b.pointIdx_sorted, H,W, 
						g.means2D,g.colors,g.invcoc2D_opacity,
						img.alpha, img.num_contributor, res);


	return num_rendered;

}
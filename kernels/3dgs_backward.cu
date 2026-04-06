#include <3dgs_backward.h>
#include <3dgs_forward.h>
#include <3dgs_structure.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <fstream>
#include <cuda.h>

namespace cg=cooperative_groups;

__global__ void BackSplat(
							uint2* ranges, uint32_t* sorted_point_idx,
							int W,int H,
							float* background,
							float2* means2D,
							float4* cov2dinv_opacity,
							float* colors,
							float* Ts,
							uint32_t* num_contributor,
							const float* dO,
							float2* d_means2D,
							float4* d_conv2Dinv,
							float* 	d_opacity,
							float* d_color	)
{
	int i=0;
	int j=0;
	int k=0;
	
	cg::thread_block block=cg::this_thread_block();
	uint32_t x_blocks=(W+BLOCK_X-1)/BLOCK_X;
	//这个block在图像中负责哪些像素点
	uint2 pix_min={block.group_index().x*BLOCK_X,block.group_index().y*BLOCK_Y};
	uint2 pix_max={min(W,block.group_index().x*BLOCK_X+BLOCK_X),min(H,block.group_index().y*BLOCK_Y+BLOCK_Y)};

	uint2 pix={pix_min.x+block.thread_index().x, pix_min.y+ block.thread_index().y};
	uint32_t pix_id=pix.y*W+pix.x;
	
	float2 pixf={(float)pix.x,(float)pix.y};

	bool done=!(pix.x<W && pix.y<H);
	bool inside=(pix.x<W && pix.y<H);
	float dL_dout[3]={0.0f};
	if(inside)
	{
		for(k=0;k<3;k++)
		{	
			dL_dout[k]=dO[k*H*W+pix_id];
		}
	}
	
	//读取这个block应该处理的Gaussian的序号
	uint2 range=ranges[block.group_index().y*x_blocks+block.group_index().x];
	
	//一个thread至多跑多少轮
	const int rounds=((range.y-range.x+BLOCK_SIZE-1)/BLOCK_SIZE);
	int toDo=range.y-range.x;
	
	//共享内存
	__shared__ int pid[BLOCK_SIZE];
	__shared__ float2 coordinates[BLOCK_SIZE];
	__shared__ float4 cov2dinv_and_opacity[BLOCK_SIZE];
	__shared__ float collected_color[3*BLOCK_SIZE];

	//按照论文的公式(2)和(3)完成渲染
	const float final_T=inside? Ts[pix_id]:0.0f;
	
	uint32_t contributor=toDo;
	uint32_t last_contributor=inside?num_contributor[pix_id]:0;
	float C[3]={0};
	
	
	float Aj[3]={0.0f};
	float Tj=final_T;
	float last_color[3]={0.0f};
	float alpha_puls=0.0f;
	for(i=0;i<rounds;i++,toDo-=BLOCK_SIZE)
	{
		block.sync();
		int current_start=i*BLOCK_SIZE;
		//如果没越界，就往共享内存读数据
		if(range.x+current_start+block.thread_rank()<range.y)
		{
			//倒着读
			int position_idx=sorted_point_idx[range.y-1-current_start-block.thread_rank()];
			pid[block.thread_rank()]=position_idx;
			coordinates[block.thread_rank()]=means2D[position_idx];
			cov2dinv_and_opacity[block.thread_rank()]=cov2dinv_opacity[position_idx];
			for(k=0;k<3;k++)
			{
				 collected_color[k*BLOCK_SIZE+block.thread_rank()]=colors[3*position_idx+k];
			}
			
		}
		block.sync();
		for(j=0;!done && j<min(BLOCK_SIZE,toDo);j++)
		{
			float dL_dalpha=0.0f;
			
			contributor-=1;
			if(contributor>=last_contributor)
			{
				continue;
			}
			float2 xy=coordinates[j];
			float2 dist={xy.x-pixf.x,xy.y-pixf.y};
			float4 cov=cov2dinv_and_opacity[j];
			
			//计算Gaussian的指数
			float exponent=-0.5*(cov.x*dist.x*dist.x+cov.z*dist.y*dist.y)-cov.y*dist.x*dist.y;
			if(exponent>0.0f)
			{
				continue;
			}

			float alphaj=min(0.99f,cov.w*exp(exponent));
			
			if(alphaj<1.0f/255.0f)
			{
				continue;
			}

			//现在开始backward运算
			Tj/=(1.0f-alphaj);
			if(Tj<0.0001f)
			{
				continue;
			}
			for(k=0;k<3;k++)
			{
				
				Aj[k]=last_color[k]*alpha_puls+(1.0-alpha_puls)*Aj[k];
				last_color[k]=collected_color[k*BLOCK_SIZE+j];

				//加上关于背景的偏微分
				dL_dalpha+=Tj*(last_color[k]-Aj[k])*dL_dout[k] - dL_dout[k]*final_T/(1.0f-alphaj)*background[k];
				atomicAdd(&(d_color[3*pid[j]+k]),dL_dout[k]*alphaj*Tj);
			}
			
			alpha_puls=alphaj;
			
			
			
			//有了关于alpha的偏微分 就能够确定关于不透明度，means2D和cov2Dinv的偏微分

			float g=exp(exponent);
			
			float dexponent_dx=-cov.x*dist.x-cov.y*dist.y;
			float dexponent_dy=-cov.z*dist.y-cov.y*dist.x;

			atomicAdd(&(d_opacity[pid[j]]),g*dL_dalpha);

			atomicAdd(&(d_means2D[pid[j]].x),-dL_dalpha*cov.w*g*dexponent_dx);
			atomicAdd(&(d_means2D[pid[j]].y),-dL_dalpha*cov.w*g*dexponent_dy);

			//关于Sigma^{-1}的偏微分
			//关于左上角
			atomicAdd(&(d_conv2Dinv[pid[j]].x),-0.5f*g*dist.x*dist.x*dL_dalpha*cov.w);
			//关于右上角和左下角(对称)
			atomicAdd(&(d_conv2Dinv[pid[j]].y),-0.5f*g*dist.x*dist.y*dL_dalpha*cov.w);
			//关于右下角
			atomicAdd(&(d_conv2Dinv[pid[j]].w),-0.5f*g*dist.y*dist.y*dL_dalpha*cov.w);
		}
	}
}


//计算dSigma^{-1}/dSigma 并把梯度传播到means3D（3D位置会对2D协方差有影响，因为计算2D协方差的时候用到了投影近似）
__global__ void ComputeCov2D(
							int num_points,float3* means3D,const int* radius,float* cov3Ds,const float focal_x, const float focal_y,
							const float tan_fovx,const float tan_fovy,
							const float* view_matrix, const float4* dL_dcov2Dinv,float3* dL_dmeans, float* dL_dcov3D)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_points || radius[idx]==0.0f)
	{
		return;
	}
	//获得当前3D协方差矩阵的起始地址，注意储存的时候每个矩阵只存右上角的6个元素
	const float* cov3D=cov3Ds+6*idx;

	float3 position=means3D[idx];
	float3 dL_dcov={dL_dcov2Dinv[idx].x,dL_dcov2Dinv[idx].y,dL_dcov2Dinv[idx].w};

	float3 cam_coor={	view_matrix[0]*position.x+view_matrix[1]*position.y+view_matrix[2]*position.z+view_matrix[3],
						view_matrix[4]*position.x+view_matrix[5]*position.y+view_matrix[6]*position.z+view_matrix[7],
						view_matrix[8]*position.x+view_matrix[9]*position.y+view_matrix[10]*position.z+view_matrix[11]};
	float tx=cam_coor.x;
	float ty=cam_coor.y;
	float tz=cam_coor.z;
	
	int i;

	const float limx=1.3f*tan_fovx;
	const float limy=1.3f*tan_fovy;
	float xdevidez=tx/tz;
	float ydevidez=ty/tz;
	tx=(float)min(limx,max(xdevidez,-limx))*tz;
	ty=(float)min(limy,max(ydevidez,-limy))*tz;

	//被剪裁掉的tx和ty不参与梯度传播
	const float grad_x_coeff=xdevidez<-limx||xdevidez>limx? 0.0f:1.0f;
	const float grad_y_coeff=ydevidez<-limx||ydevidez>limy? 0.0f:1.0f;


	//实际上J的最后一行不是0，但我们最终的计算结果只取三维矩阵的左上二阶分块, 这个二阶分块是不受J的第三行影响的.
	mat3 J={
			focal_x/tz,0,-1.0f*focal_x*tx/(tz*tz),
			0,focal_y/tz,-1.0f*focal_y*ty/(tz*tz),
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

	mat3 T=matmul(J,W);
	mat3 cov2D=matmul(T,matmul(sigma,transpose(T)));



	float a=cov2D.data[0]+0.3f;
	float b=cov2D.data[1];
	float c=cov2D.data[4]+0.3f;
	float det=a*c-b*b;

	float deno=1.0f/(det*det+0.0000001f);
	float dL_da,dL_db,dL_dc;

	if(deno!=0)
	{
		//dL_dcov是关于Sigma2D^{-1}的偏微分, 现在计算L关于Sigma2D的偏微分
		dL_da=deno*(-c*c*dL_dcov.x+2.0f*b*c*dL_dcov.y+(det-a*c)*dL_dcov.z);
		dL_dc=deno*(-a*a*dL_dcov.z+2.0f*a*b*dL_dcov.y+(det-a*c)*dL_dcov.x);

		dL_db=deno*2.0f*(b*c*dL_dcov.x-(det+2.0f*b*b)*dL_dcov.y+a*b*dL_dcov.z);
		

		//有了dL_dcov2D, 可以计算 dL_dcov3D
		dL_dcov3D[6*idx]=(T[0]*T[0]*dL_da+T[0]*T[3]*dL_db+T[3]*T[3]*dL_dc);
		dL_dcov3D[6*idx+3]=T[1]*T[1]*dL_da+T[1]*T[4]*dL_db+T[4]*T[4]*dL_dc;
		dL_dcov3D[6*idx+5]=T[2]*T[2]*dL_da+T[2]*T[5]*dL_db+T[5]*T[5]*dL_dc;

		dL_dcov3D[6*idx+1]=2*T[0]*T[1]*dL_da+(T[0]*T[4]+T[1]*T[3])*dL_db+2*T[3]*T[4]*dL_dc;
		dL_dcov3D[6*idx+2]=2*T[0]*T[2]*dL_da+(T[0]*T[5]+T[2]*T[3])*dL_db+2*T[3]*T[5]*dL_dc;
		dL_dcov3D[6*idx+4]=2*T[1]*T[2]*dL_da+(T[1]*T[5]+T[2]*T[4])*dL_db+2*T[4]*T[5]*dL_dc;

		
	}
	else
	{
		for(i=0;i<6;i++)
		{
			dL_dcov3D[6*idx+i]=0.0f;
		}
	}

	//然后计算dL_dT, 因为T是由tx ty tz决定的
	float dL_dT00=2*(T(0,0)*sigma(0,0)+T(0,1)*sigma(0,1)+T(0,2)*sigma(0,2))*dL_da+(T(1,0)*sigma(0,0)+T(1,1)*sigma(0,1)+T(1,2)*sigma(0,2))*dL_db;
	float dL_dT01=2*(T(0,0)*sigma(1,0)+T(0,1)*sigma(1,1)+T(0,2)*sigma(1,2))*dL_da+(T(1,0)*sigma(1,0)+T(1,1)*sigma(1,1)+T(1,2)*sigma(1,2))*dL_db;

	float dL_dT02=2*(T(0,0)*sigma(2,0)+T(0,1)*sigma(2,1)+T(0,2)*sigma(2,2))*dL_da+(T(1,0)*sigma(2,0)+T(1,1)*sigma(2,1)+T(1,2)*sigma(2,2))*dL_db;

	float dL_dT10=2*(T(1,0)*sigma(0,0)+T(1,1)*sigma(0,1)+T(1,2)*sigma(0,2))*dL_dc+(T(0,0)*sigma(0,0)+T(0,1)*sigma(0,1)+T(0,2)*sigma(0,2))*dL_db;
	float dL_dT11=2*(T(1,0)*sigma(1,0)+T(1,1)*sigma(1,1)+T(1,2)*sigma(1,2))*dL_dc+(T(0,0)*sigma(1,0)+T(0,1)*sigma(1,1)+T(0,2)*sigma(1,2))*dL_db;
	float dL_dT12=2*(T(1,0)*sigma(2,0)+T(1,1)*sigma(2,1)+T(1,2)*sigma(2,2))*dL_dc+(T(0,0)*sigma(2,0)+T(0,1)*sigma(2,1)+T(0,2)*sigma(2,2))*dL_db;


	//dL_dJ
	float dL_dJ00=W(0,0)*dL_dT00+W(0,1)*dL_dT01+W(0,2)*dL_dT02;
	float dL_dJ02=W(2,0)*dL_dT00+W(2,1)*dL_dT01+W(2,2)*dL_dT02;

	float dL_dJ11=W(1,0)*dL_dT10+W(1,1)*dL_dT11+W(1,2)*dL_dT12;
	float dL_dJ12=W(2,0)*dL_dT10+W(2,1)*dL_dT11+W(2,2)*dL_dT12;

	
	tz=1.0f/tz;
	float tz2=tz*tz;
	float tz3=tz2*tz;

	float dL_dtx=-grad_x_coeff*focal_x*tz2*dL_dJ02;
	float dL_dty=-grad_y_coeff*focal_y*tz2*dL_dJ12;
	float dL_dtz=-focal_x*tz2*dL_dJ00-focal_y*tz2*dL_dJ11+2.0f*(focal_x*tx)*tz3*dL_dJ02+2.0f*(focal_y*ty)*tz3*dL_dJ12;

	position={dL_dtx,dL_dty,dL_dtz};
	//这个时候微分是转置 而且做的是向量变换， 因此齐次坐标最后一维是0
	dL_dmeans[idx]={	view_matrix[0]*position.x+view_matrix[4]*position.y+view_matrix[8]*position.z,
						view_matrix[1]*position.x+view_matrix[5]*position.y+view_matrix[9]*position.z,
						view_matrix[2]*position.x+view_matrix[6]*position.y+view_matrix[10]*position.z};

}

__device__ void ComputeCov3D(int idx, float3 scale,float scale_modifier,float4 quant_num, float* dL_dcov3D, float* dL_dscale, float* dL_dquant)
{
	//旋转矩阵
	
	float r=quant_num.x;
	float i=quant_num.y;
	float j=quant_num.z;
	float k=quant_num.w;
	// row major
	mat3 R={
			2*(0.5-j*j-k*k),
			2*(i*j-r*k),
			2*(i*k+r*j),
			2*(i*j+r*k),
			2*(0.5-i*i-k*k),
			2*(j*k-r*i),
			2*(i*k-r*j),
			2*(j*k+r*i),
			2*(0.5-i*i-j*j)};

	mat3 S={ scale_modifier* scale.x,0.0f,0.0f,
			0.0f,scale_modifier* scale.y,0.0f,
			0.0f,0.0f,scale_modifier* scale.z};

	mat3 M=matmul(R,S);
	mat3 dL_dSigma={
					dL_dcov3D[0],0.5f*dL_dcov3D[1],0.5f*dL_dcov3D[2],
					0.5f*dL_dcov3D[1],dL_dcov3D[3],0.5f*dL_dcov3D[4],
					0.5f*dL_dcov3D[2],0.5f*dL_dcov3D[4],dL_dcov3D[5]};
	mat3 dL_dM=2.0f*matmul(dL_dSigma,M);
	mat3 dL_dMT=transpose(dL_dM);
	mat3 RT=transpose(R);

	dL_dscale[0]=RT(0,0)*dL_dM(0,0)+RT(0,1)*dL_dM(1,0)+RT(0,2)*dL_dM(2,0);
	dL_dscale[1]=RT(1,0)*dL_dM(0,1)+RT(1,1)*dL_dM(1,1)+RT(1,2)*dL_dM(2,1);
	dL_dscale[2]=RT(2,0)*dL_dM(0,2)+RT(2,1)*dL_dM(1,2)+RT(2,2)*dL_dM(2,2);

	//按照论文的公式（11）完成对四元数的微分运算. 注意这些矩阵还没有乘以2和scale_modifier!
	mat3 dM_dr={ 0.0f,-scale.y*k,scale.z*j,
				scale.x*k,0.0f,-scale.z*i,
				-scale.x*j,scale.y*i,0.0f};
	mat3 dM_di={
				0.0f,scale.y*j,scale.z*k,
				scale.x*j,-2.0f*scale.y*i,-scale.z*r,
				scale.x*k,scale.y*r,-2.0f*scale.z*i};
	mat3 dM_dj={
				-2.0f*scale.x*j,scale.y*i,scale.z*r,
				scale.x*i,0.0f,scale.z*k,
				-scale.x*r,scale.y*k,-2.0f*scale.z*j};
	mat3 dM_dk={
				-2.0f*scale.x*k,-scale.y*r,scale.z*i,
				scale.x*r,-2.0f*scale.y*k,scale.z*j,
				scale.x*i,scale.y*j,0.0f};

	float4 dL_dq={0.0f,0.0f,0.0f,0.0f};
	dL_dq.x=mat_dot(dL_dM, dM_dr)*2.0f*scale_modifier;
	dL_dq.y=mat_dot(dL_dM, dM_di)*2.0f*scale_modifier;
	dL_dq.z=mat_dot(dL_dM, dM_dj)*2.0f*scale_modifier;
	dL_dq.w=mat_dot(dL_dM, dM_dk)*2.0f*scale_modifier;

	dL_dquant[0]=mat_dot(dL_dM, dM_dr)*2.0f*scale_modifier;
	dL_dquant[1]=mat_dot(dL_dM, dM_di)*2.0f*scale_modifier;
	dL_dquant[2]=mat_dot(dL_dM, dM_dj)*2.0f*scale_modifier;
	dL_dquant[3]=mat_dot(dL_dM, dM_dk)*2.0f*scale_modifier;

}

__global__ void ComputeMeans3D(	
								int num_points,
								const int* radius,
								const float3* means,
								float3* scales,
								float4* quant_number,
								const float scale_modifier,
								const float* proj_matrix,
								const float2* dL_dmeans2D,
								float* dL_dmeans,
								float* dL_dcov3D,
								float* dL_dscale,
								float* dL_dq)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_points|| radius[idx]<=0)
	{
		return;
	}

	float3 position=means[idx];
	float mx=proj_matrix[0]*position.x+proj_matrix[1]*position.y+proj_matrix[2]*position.z+proj_matrix[3];
	float my=proj_matrix[4]*position.x+proj_matrix[5]*position.y+proj_matrix[6]*position.z+proj_matrix[7];
	float w=proj_matrix[12]*position.x+proj_matrix[13]*position.y+proj_matrix[14]*position.z+proj_matrix[15];

	//变成倒数
	w=1.0f/(w+0.0000001f);
	float w2=w*w;
	float xx=proj_matrix[0]*w-proj_matrix[12]*mx*w2;
	float xy=proj_matrix[1]*w-proj_matrix[13]*mx*w2;
	float xz=proj_matrix[2]*w-proj_matrix[14]*mx*w2;

	float yx=proj_matrix[4]*w-proj_matrix[12]*my*w2;
	float yy=proj_matrix[5]*w-proj_matrix[13]*my*w2;
	float yz=proj_matrix[6]*w-proj_matrix[14]*my*w2;

	float dL_dmean2Dx=dL_dmeans2D[idx].x;
	float dL_dmean2Dy=dL_dmeans2D[idx].y;
	//加上关于cov2D时传播的梯度
	dL_dmeans[idx*3]+=dL_dmean2Dx*xx+dL_dmean2Dy*yx;
	dL_dmeans[idx*3+1]+=dL_dmean2Dx*xy+dL_dmean2Dy*yy;
	dL_dmeans[idx*3+2]+=dL_dmean2Dx*xz+dL_dmean2Dy*yz;

	float3 s=scales[idx];
	float4 q=quant_number[idx];

	ComputeCov3D(idx, s,scale_modifier,q, dL_dcov3D+6*idx, dL_dscale+3*idx, dL_dq+4*idx);


}

void gsBackward(
				char* geometryBuffer,
				char* binningBuffer,
				char* imageBuffer,
				int H,int W,
				int num_points,
				int num_rendered,
				float* background,
				float* means3D,
				float* colors,
				float* scales,
				const float scale_modifier,
				float* quant_number,
				float* view_matrix,
				float* proj_matrix,
				float tan_fovx,
				float tan_fovy,
				const float* dL_dres,
				float* dL_dmeans2D,
				float* dL_dcov2Dinv,
				float* dL_dopacity,
				float* dL_dcolor,
				float* dL_dmeans3D,
				float* dL_dcov3D,
				float* dL_dscale,
				float* dL_dq)
{
	
	Geometry g=Geometry::fromChunk(geometryBuffer,num_points);
	Binning b=Binning::fromChunk(binningBuffer,num_rendered);
	Image img=Image::fromChunk(imageBuffer,H*W);


	const float focal_x=W/(2.0f*tan_fovx);
	const float focal_y=H/(2.0f*tan_fovy);

	dim3 grid_dim((W+BLOCK_X-1)/BLOCK_X,(H+BLOCK_Y-1)/BLOCK_Y,1);
	dim3 block_dim(BLOCK_X,BLOCK_Y,1);



	BackSplat<<<grid_dim,block_dim>>>(
							img.ranges, b.pointIdx_sorted,
							W,H,
							background,
							g.means2D,
							g.invcoc2D_opacity,
							colors,
							img.alpha,
							img.num_contributor,
							dL_dres,
							(float2*) dL_dmeans2D,
							(float4*) dL_dcov2Dinv,
							dL_dopacity,
							dL_dcolor	);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("BackSplat核函数启动失败: %s\n", cudaGetErrorString(err));
		return;
	}
	
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("BackSplat核函数执行失败: %s\n", cudaGetErrorString(err));
		return;
	}

	


	ComputeCov2D<<<(num_points+255)/256,256>>>(num_points,(float3*)means3D,g.radius,g.cov3D,
							focal_x,focal_y,tan_fovx,tan_fovy,
							view_matrix, (float4*) dL_dcov2Dinv,(float3*) dL_dmeans3D, dL_dcov3D);
	
	ComputeMeans3D<<<(num_points+255)/256,256>>>(	
								num_points,
								g.radius,
								(const float3*)means3D,
								(float3*)scales,
								(float4*)quant_number,
								scale_modifier,
								proj_matrix,
								(const float2*) dL_dmeans2D,
								dL_dmeans3D,
								dL_dcov3D,
								dL_dscale,
								dL_dq);

}
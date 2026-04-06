#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdio.h>
#include <iostream>
#include <functional>
#include <cub/cub.cuh>
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X*BLOCK_Y)
#define EPS 0.00001
struct mat3
{
	float data[9];

	__host__ __device__ float& operator[](int index)
	{
		return data[index];
	}

	__host__ __device__ float& operator()(int i, int j)
	{
		return data[3 * i + j];
	}

	 __host__ __device__ const float& operator[](int index) const
	{
		return data[index];
	}
};

__host__ __device__ inline mat3 operator*(float scalar, const mat3& m)
{
	mat3 res;
	for (int i = 0; i < 9; i++)
	{
		res[i] = scalar * m[i];
	}
	return res;
}

__host__ __device__ inline float mat_dot(const mat3& m1, const mat3& m2)
{
	float sum = 0.0f;
	int i = 0;
	for (i = 0; i < 9; i++)
	{
		sum += m1[i] * m2[i];
	}
	return sum;
}


__host__ __device__ inline mat3 transpose(const mat3& M)
{	
	mat3 Mt = {
			M.data[0],M.data[3],M.data[6],
			M.data[1],M.data[4],M.data[7],
			M.data[2],M.data[5],M.data[8] };
	return Mt;
}

__host__ __device__ inline mat3 matmul(const mat3& A, const mat3& B)
{
	mat3 R = {
		A.data[0] * B.data[0] + A.data[1] * B.data[3] + A.data[2] * B.data[6], A.data[0] * B.data[1] + A.data[1] * B.data[4] + A.data[2] * B.data[7],A.data[0] * B.data[2] + A.data[1] * B.data[5] + A.data[2] * B.data[8],
		A.data[3] * B.data[0] + A.data[4] * B.data[3] + A.data[5] * B.data[6], A.data[3] * B.data[1] + A.data[4] * B.data[4] + A.data[5] * B.data[7],A.data[3] * B.data[2] + A.data[4] * B.data[5] + A.data[5] * B.data[8],
		A.data[6] * B.data[0] + A.data[7] * B.data[3] + A.data[8] * B.data[6], A.data[6] * B.data[1] + A.data[7] * B.data[4] + A.data[8] * B.data[7],A.data[6] * B.data[2] + A.data[7] * B.data[5] + A.data[8] * B.data[8]
	};
	return R;
}

inline __forceinline__ __device__ void getRect(const float2 p, int radius, uint2& left_bottom, uint2& right_up, dim3 grid)
{
	left_bottom = {
						min(grid.x,max((int)0,(int)((p.x - radius) / BLOCK_X))),min(grid.y,max((int)0,(int)((p.y - radius) / BLOCK_Y))) };
	right_up = {
						min(grid.x,max((int)0,(int)((p.x + radius+BLOCK_X-1) / BLOCK_X))),min(grid.y,max((int)0,(int)((p.y + radius+BLOCK_Y-1) / BLOCK_Y))) };
}

//对每个高斯点进行预处理(3D投影到相机屏幕)
void preprocess(int num_points, const float* means3D, const float* scales, const float scale_modifier, const float* quant_number,
	const float* opacity,  const float* view_matrix, const float* project_matrix, const float* camera_position,
	const int H, const int W, const float focal_x, const float focal_y, const float tan_fovx, const float tan_fovy, int* radius, float2* means2D, float* depth,
	float* cov3D, float* color, float4* conic_opacity, const dim3 grid, uint32_t* insected_tiles);

//主函数
int gsForward(std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binnBuffer,
	std::function<char* (size_t)> ImgBuffer,
	const float* background,
	int num_points,
	const float* means3D,
	const float* colors,
	const float* scale,
	const float scale_modifier,
	const float* quant_number,
	const float* opacity,
	const float* view_matrix,
	const float* proj_matrix,
	const int H,
	const int W,
	const float tan_fovx,
	const float tan_fovy,
	float* res);


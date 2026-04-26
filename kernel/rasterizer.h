#pragma once
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

template <typename T>
static void manual_allocate(char*& address, T*& ptr, std::size_t count, std::size_t alignment)
{
	/*
		输入:
			address 内存池当前剩余空间的起始地址
			ptr T类型的指针， 分配成功后指向分配后的地址的起点
			count 要分配的T的数目
			alignment 对齐的字节数, 2的幂次
	*/

	//把地址chunk变成无符号整数，然后增加到一个上界，再把低位的余量剪掉.
	std::size_t offset = (reinterpret_cast<std::uintptr_t>(address) + alignment - 1) & ~(alignment - 1);

	//把分配后的地址转化为T*类型, 再递给ptr
	ptr = reinterpret_cast<T*>(offset);
	//更新可用内存的起始地址
	address = reinterpret_cast<char*>(offset + count * sizeof(T));
}
//传入空指针模拟内存分配的过程 完全不会访问内存 只是计算偏移量
template <typename T>
size_t PrecomputeBias(size_t N)
{
	char* size = nullptr;
	T::ALLOC(size, N);
	return ((size_t)size);
}

struct GeometryInfo
{

	float* depth;               //顶点的深度值数组
	float2* screen_coor;        //顶点的屏幕坐标数组
	
	static GeometryInfo ALLOC(char*& address, size_t N);

};
struct IntermediateInfo
{
	uint32_t* tiles_intersected;         //每个三角形和多少个tile相交.
	uint32_t* scanned_res;
	size_t scan_size;
	char* scan_address;                  //前缀和地址

	static IntermediateInfo ALLOC(char*& address, size_t N);
};

struct TriangleInfo
{
	size_t sort_size;
	uint32_t* keysUnsorted;              //存放未排序的键值
	uint32_t* keysSorted;                //存放排序的键值
	uint32_t* TriangleIndexUnsorted;     //存放未排序的三角形索引
	uint32_t* TriangleIndexSorted;		//存放排序的三角形索引

	char* sortAddress;                 //排序地址



	static TriangleInfo ALLOC(char*& address, size_t N);

};

struct ImageInfo
{
	uint2* ranges;

	static ImageInfo ALLOC(char*& address, size_t N);
};

__device__ inline float3 ScreenProject(const float* proj_matrix, const float3 point)
{
	
	float screen_x = proj_matrix[0] * point.x + proj_matrix[1] * point.y + proj_matrix[2] * point.z + proj_matrix[3];
	float screen_y= proj_matrix[4] * point.x + proj_matrix[5] * point.y + proj_matrix[6] * point.z + proj_matrix[7];

	float w= proj_matrix[12] * point.x + proj_matrix[13] * point.y + proj_matrix[14] * point.z + proj_matrix[15];

	float inv = 1.0f / w;
	float3 res = { screen_x * inv, screen_y * inv, w };
	return res;

}

__device__ inline void AABB(float2 v1, float2 v2, float2 v3, uint2& left_bottom, uint2& right_up,dim3 grid)
{
	//先获得这三个顶点的bounding box
	float right = max(v1.x, max(v2.x, v3.x));
	float left= min(v1.x, min(v2.x, v3.x));
	float up= max(v1.y, max(v2.y, v3.y));
	float bottom= min(v1.y, min(v2.y, v3.y));

	left_bottom = { min(grid.x, max((int)0,(int)(left/BLOCK_X)-1) ),  min(grid.y, max((int)0,(int)(bottom / BLOCK_Y)-1)) };

	right_up= { min(grid.x, max((int)0,1+(int)( (right+BLOCK_X-1) / BLOCK_X))),  min(grid.y,1+ max((int)0,(int)( (up+BLOCK_Y-1) / BLOCK_Y))) };

}

__device__ inline bool CheckTriangle(float2 v1, float2 v2, float2 v3, float2 p, float& b1, float& b2)
{
	//检查p在不在二维的三角形里 如果在就把三线性插值的v1和v2权重放入b1 b2中


	float s123 = abs((v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x));
	//float s12p = abs((v2.x - v1.x) * (p.y - v1.y) - (v2.y - v1.y) * (p.x - v1.x));
	float s23p= abs((v2.x - p.x) * (v3.y - p.y) - (v2.y - p.y) * (v3.x - p.x));
	float s13p= abs((p.x - v1.x) * (v3.y - v1.y) - (p.y - v1.y) * (v3.x - v1.x));

	float flag1 = ((v3.x - v1.x) * (v2.y - v1.y) - (v2.x - v1.x) * (v3.y - v1.y)) * ((v3.x - v1.x) * (p.y - v1.y) - (p.x - v1.x) * (v3.y - v1.y));
	float flag2 = ((v2.x - v3.x) * (v1.y - v3.y) - (v1.x - v3.x) * (v2.y - v3.y)) * ((v2.x - v3.x) * (p.y - v3.y) - (p.x - v3.x) * (v2.y - v3.y));
	float flag3 = ((v1.x - v2.x) * (v3.y - v2.y) - (v3.x - v2.x) * (v1.y - v2.y)) * ((v1.x - v2.x) * (p.y - v2.y) - (p.x - v2.x) * (v1.y - v2.y));


	float flag4= ((v2.x - v1.x) * (v3.y - v1.y) - (v3.x - v1.x) * (v2.y - v1.y)) * ((v3.x - v1.x) * (p.y - v1.y) - (p.x - v1.x) * (v3.y - v1.y));
	float flag5= ((v1.x - v3.x) * (v2.y - v3.y) - (v2.x - v3.x) * (v1.y - v3.y)) * ((v2.x - v3.x) * (p.y - v3.y) - (p.x - v3.x) * (v2.y - v3.y));
	float flag6= ((v3.x - v2.x) * (v1.y - v2.y) - (v1.x - v2.x) * (v3.y - v2.y)) * ((v1.x - v2.x) * (p.y - v2.y) - (p.x - v2.x) * (v1.y - v2.y));

	if ( (flag1>=0 && flag2>= 0 &&flag3>=0)|| (flag4 <= 0&& flag5 <= 0 && flag6 <= 0))
	{
		b1 = s23p / s123;
		b2 = s13p / s123;
		
		return true;
	}
	return false;
}

void RasterizationFWD(
	std::function<char* (size_t)> geometryInfo,
	std::function<char* (size_t)> intermediateInfo,
	std::function<char* (size_t)> triangleInfo,
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
	float* res);


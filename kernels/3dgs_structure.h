#pragma once

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
//对于性能要求极高的场合, 我们手动分配内存
template <typename T>
static void manual_allocate(char*& address, T*& ptr,std::size_t count, std::size_t alignment)
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
size_t required(size_t N)
{
	char* size = nullptr;
	T::fromChunk(size, N);
	return ((size_t)size);
}
//Geometry用于存放关于Gaussian的信息, 比如点的数目, 3D位置, 2D协方差等等
struct  Geometry
{
	float2* means2D;
	float* colors;
	float* opacity;
	float* depth;
	float* cov3D;
	float4* invcoc2D_opacity;
	int* radius;
	size_t scan_size;
	uint32_t* scanned_points;
	uint32_t* insected_tiles;
	char* scan_address;



	static Geometry fromChunk(char*& chunk, size_t N);

};

struct Binning
{
	size_t sort_size;
	uint64_t* keys_unsorted;
	uint64_t* keys_sorted;
	uint32_t* pointsIdx_unsorted;
	uint32_t* pointIdx_sorted;
	char* list_sorting_space;

	static Binning fromChunk(char*& chunk, size_t N);
};

struct Image
{
	uint2* ranges;
	uint32_t* num_contributor;
	float* alpha;

	static Image fromChunk(char*& chunk, size_t N);
};
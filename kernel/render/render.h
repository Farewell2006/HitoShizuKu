#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdio.h>
#include <iostream>
#include <functional>
#include <cub/cub.cuh>
#include <math_utils.h>


#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X*BLOCK_Y)




template <typename T>
class Bounds3
{
public:
	T minNum = -1e30f;
	T maxNum = 1e30f;
	vector3<T> pMin = vector3<T>(minNum, minNum, minNum);
	vector3<T> pMax = vector3<T>(maxNum, maxNum, maxNum);
	__host__ __device__ Bounds3() = default;

	__host__ __device__ Bounds3(const vector3<T>& p)
	{
		pMin=p;
		pMax=p;
	}

	__host__ __device__ vector3<T> Diagonal()const { return pMax - pMin; }
	__host__ __device__ T Volume()const {
		vector3<T> d = Diagonal();
		return d.x * d.y * d.z;
	}

	__host__ __device__ T surfaceArea()const {
		vector3<T> d = Diagonal();
		return 2*(d.x*d.y+d.z * d.y +d.x* d.z);
	}

	__host__ __device__ int MaximumExtent()const {
		vector3<T> d = Diagonal();
		if (d.x >= d.y && d.x >= d.z)
		{
			return 0;
		}
		else if (d.y >= d.x && d.y >= d.z)
		{
			return 1;
		}
		else
		{
			return 2;
		}
	}

	__host__ __device__ vector3<T> centroid()const {
		vector3<T> d = Diagonal();
		return 0.5f * d + pMin;
	}

	

};

template <typename T> 
__host__ __device__ Bounds3<T> Union(Bounds3<T>& b, const vector3<T>& p)
{
	Bounds3<T> b1;
	b1.pMin = vector3<T>(std::min(b.pMin.x, p.x), std::min(b.pMin.y, p.y), std::min(b.pMin.z, p.z));
	b1.pMax = vector3<T>(std::max(b.pMax.x, p.x), std::max(b.pMax.y, p.y), std::max(b.pMax.z, p.z));
	return b1;
};

template <typename T> 
__host__ __device__ Bounds3<T> Union(Bounds3<T>& b, Bounds3<T>& p)
{
	Bounds3<T> b1;
	b1.pMin = vector3<T>(std::min(b.pMin.x, p.pMin.x), std::min(b.pMin.y, p.pMin.y), std::min(b.pMin.z, p.pMin.z));
	b1.pMax = vector3<T>(std::max(b.pMax.x, p.pMax.x), std::max(b.pMax.y, p.pMax.y), std::max(b.pMax.z, p.pMax.z));

	return b1;
};

typedef Bounds3<float> Bounds3f;




__device__ inline vec3f ProjectToWorld(vec3f point, const float* cam_to_world)
{
	vec3f res;
	res.x = cam_to_world[0] * point.x + cam_to_world[1] * point.y + cam_to_world[2] * point.z + cam_to_world[3];
	res.y = cam_to_world[4] * point.x + cam_to_world[5] * point.y + cam_to_world[6] * point.z + cam_to_world[7];
	res.z = cam_to_world[8] * point.x + cam_to_world[9] * point.y + cam_to_world[10] * point.z + cam_to_world[11];
	return res;
};

struct Material
{
	int texture_id;
};



struct Primitive
{
	vec3f points[3];
	vec2f uv[3];
	int material_id;
	Bounds3f BBox;
	int idx;
	

};

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


struct Scene
{
	int num_primitives;
	int num_materials;
	Material* materials;
	Primitive* primitives;



	static Scene ALLOC(char*& address, size_t num_primitives, size_t num_materials);
};



//传入空指针模拟内存分配的过程 完全不会访问内存 只是计算偏移量
static size_t PrecomputeSceneBias(size_t num_primitives, size_t num_materials)
{
	char* size = nullptr;
	Scene::ALLOC(size, num_primitives, num_materials);
	return ((size_t)size);
};

struct Ray
{
	vec3f dir;
	vec3f ori;
	vec3f color;

	float tMax;

	int pix_id;
};

struct RayBuffer
{
	Ray* rays;
	int num_rays;

	static RayBuffer ALLOC(char*& address, size_t num_pixels);
};


static size_t PrecomputeRayBufferBias(size_t num_pixels)
{
	char* size = nullptr;
	RayBuffer::ALLOC(size, num_pixels);
	return ((size_t)size);
};

struct Interaction
{
	bool intersected;
	vec3f point;
	vec2f uv;

	int material_id;
	int ray_idx;
};

struct InteractionBuffer
{
	int num_interactions;
	Interaction* interactions;

	static InteractionBuffer ALLOC(char*& address, size_t num_interactions);
};

static size_t PrecomputeInteractionBufferBias(size_t num_interactions)
{
	char* size = nullptr;
	InteractionBuffer::ALLOC(size, num_interactions);
	return ((size_t)size);

};


__device__ inline bool CheckInsect(const Bounds3f& b, const Ray& ray, const vec3f& invDir, const int DirIsNeg[3])
{
	float tMin = (b.pMin.x - ray.ori.x) * invDir.x;
	float tMax = (b.pMax.x - ray.ori.x) * invDir.x;
	if (DirIsNeg[0] == 1)
	{
		tMin = (b.pMax.x - ray.ori.x) * invDir.x;
		tMax = (b.pMin.x - ray.ori.x) * invDir.x;
	}
	

	float tyMin = (b.pMin.y - ray.ori.y) * invDir.y;
	float tyMax = (b.pMax.y - ray.ori.y) * invDir.y;
	if (DirIsNeg[1] == 1)
	{
		tyMin = (b.pMax.y - ray.ori.y) * invDir.y;
		tyMax = (b.pMin.y - ray.ori.y) * invDir.y;
	}


	float tzMin = (b.pMin.z - ray.ori.z) * invDir.z;
	float tzMax = (b.pMax.z - ray.ori.z) * invDir.z;
	if (DirIsNeg[2] == 1)
	{
		tzMin = (b.pMax.z - ray.ori.z) * invDir.z;
		tzMax = (b.pMin.z - ray.ori.z) * invDir.z;
	}

	if (tMin > tyMax || tyMin > tMax)
	{
		return false;
	}
	if (tyMin > tMin) tMin = tyMin;
	if (tyMax < tMax) tMax = tyMax;

	if (tMin > tzMax || tzMin > tMax)
	{
		return false;
	}
	if (tzMin > tMin) tMin = tzMin;
	if (tzMax < tMax) tMax = tzMax;
	
	return (tMin < ray.tMax) && (tMax >0);

};

__device__ inline bool TriangleRayInsect(const Primitive& p, const Ray& ray, float& time, float& coeff0,float& coeff1)
{
	vec3f P0 = p.points[0];
	vec3f P1= p.points[1];
	vec3f P2 = p.points[2];

	vec3f E1 = P1 - P0;
	vec3f E2 = P2 - P0;
	vec3f S = ray.ori - P0;
	vec3f S1 = cross(ray.dir, E2);
	vec3f S2 = cross(S, E1);

	float deno = dot(S1, E1);
	if (abs(deno)==0)
	{
		return false;
	}
	else
	{
		float inv_deno = 1.0f / deno;
		float t = dot(S2, E2) * inv_deno;
		float b1 = dot(S1, S) * inv_deno;
		float b2 = dot(S2, ray.dir) * inv_deno;

		if ((t >= 0) && (b1 + b2 <= 1.0) && (b1 >= 0) && (b2 >= 0))
		{
			time = t;
			coeff0 = 1.0f - b1 - b2;
			coeff1 = b1;
			return true;
		}
	}
	return false;
};

__device__ inline void TriangleGradCompute(const Primitive& p, const Ray& ray, const float dLdb0, const float dLdb1, vec3f& dLdp0, vec3f& dLdp1, vec3f& dLdp2)
{
	vec3f P0 = p.points[0];
	vec3f P1 = p.points[1];
	vec3f P2 = p.points[2];

	vec3f E1 = P1 - P0;
	vec3f E2 = P2 - P0;
	vec3f S = ray.ori - P0;
	vec3f S1 = cross(ray.dir, E2);
	vec3f S2 = cross(S, E1);
	vec3f D = ray.dir;
	vec3f O = ray.ori;

	float deno = dot(S1, E1);
	if (abs(deno) == 0)
	{
		deno = 1.0f;
	}
	else
	{
		float inv_deno = 1.0f / deno;
		float t = dot(S2, E2) * inv_deno;
		float b1 = dot(S1, S) * inv_deno;
		float b2 = dot(S2, ray.dir) * inv_deno;

		if ((t >= 0) && (b1 + b2 <= 1.0) && (b1 >= 0) && (b2 >= 0))
		{
			vec3f db1_dp0 = (cross(D, O - P2) - b1 * (cross(D, P1 - P2))) * inv_deno;
			vec3f db1_dp1 = -b1 * cross(D, E2) * inv_deno;
			vec3f db1_dp2 = (cross(O - P0, D) - b1 * cross(E1, D)) * inv_deno;

			vec3f db2_dp0 = (cross(D, P1 - O) - b2 * cross(D, P1 - P2)) * inv_deno;
			vec3f db2_dp1 = (cross(D, O - P0) - b2 * cross(D, P2 - P0)) * inv_deno;
			vec3f db2_dp2 = -b2 * cross(P1 - P0, D) * inv_deno;

			

			dLdp0 = dLdb1 * db1_dp0 - dLdb0 *(db1_dp0+ db2_dp0);
			dLdp1 = dLdb1 * db1_dp1 -dLdb0 * (db1_dp1 + db2_dp1);
			dLdp2 = dLdb1 * db1_dp2 - dLdb0 * (db1_dp2 + db2_dp2);

		}
	}
	
}

__device__ inline bool BilinearSample(vec2f uv, int texture_H, int texture_W, int& start_x, int& start_y)
{
	int u = (int)(uv.x);
	int v = (int)(uv.y);

	u = max(0, u);
	v = max(0, v);

	u = min(texture_W - 1, u);
	v = min(texture_H - 1, v);

	start_x = u - 1;
	start_y = v - 1;
	if (u % 2 == 0)
	{
		start_x = u;
	}
	if (v % 2 == 0)
	{
		start_y = v;
	}
	return true;
}

void PathTraceFWD(
	std::function<char* (size_t)> SceneInfo,
	std::function<char* (size_t)> RayInfo,
	std::function<char* (size_t)> BBoxInfo,
	std::function<char* (size_t)> InteractionInfo,
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
	int* res_text_id);

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
		float* d_texture);
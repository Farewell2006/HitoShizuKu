#include <render.h>
#include <sah.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda.h>
#include <fstream>
#include <cub/cub.cuh>

namespace cg=cooperative_groups;


BVHNode* BuildTree(std::vector<Primitive>& primitives, int start, int end, int* total_num)
{
	int num_p = end - start;
	//只有一个图元肯定是叶子结点
	if (num_p == 1)
	{
		BVHNode* p = (BVHNode*)malloc(sizeof(BVHNode));
		p->BoundingBox = primitives[start].BBox;
		p->left = nullptr;
		p->right = nullptr;
		p->axis = 0;
		p->start = start;
		p->end = end;
		p->is_leaf = true;
		p->tri_id = primitives[start].idx;

		(*total_num) += 1;
		return p;
	}
	//否则不是叶子节点
	BVHNode* p = (BVHNode*)malloc(sizeof(BVHNode));
	p->BoundingBox = primitives[start].BBox;
	p->start = start;
	p->end = end;
	p->is_leaf = false;
	p->tri_id = -1;

	(*total_num) += 1;
	//计算该节点的BoundingBox
	int i;
	for (i = start + 1; i < end; i++)
	{
		p->BoundingBox = Union(p->BoundingBox, primitives[i].BBox);
	}
	int mid = (start + end) / 2;
	//然后开始用SAH寻找分割点,在该BoundingBox范围最大的轴上分割
	int axis = p->BoundingBox.MaximumExtent();
	//整个轴分成32份
	int split_size = 32;
	float split_point = 0.0;
	float target_split_point = 0.0;
	float width = 0.0;
	float cost = 0.0f;
	float min_cost = std::numeric_limits<float>::max();
	int na, nb;
	if (axis == 0)
	{
		cost = 0.0f;
		//沿x轴的划分
		width = p->BoundingBox.Diagonal().x;
		for (i = 0; i < split_size; i++)
		{
			split_point = p->BoundingBox.pMin.x + i * width / split_size;
			//分成x轴向<=split_point和>split_point的元素
			auto parti_pos = std::partition(primitives.begin() + start, primitives.begin() + end, [split_point](const Primitive& prim) {return prim.BBox.centroid().x <= split_point; });
			na = 0;
			nb = 0;
			Bounds3f BA = primitives[start].BBox;
			na += 1;
			for (auto iter = primitives.begin() + start; iter != parti_pos; iter++)
			{
				if (iter == primitives.begin() + start)
				{
					continue;
				}
				else
				{
					BA = Union(BA, iter->BBox);
					na += 1;
				}

			}
			Bounds3f BB = parti_pos->BBox;
			nb += 1;
			for (auto iter = parti_pos; iter != primitives.begin() + end; iter++)
			{
				if (iter == parti_pos)
				{
					continue;
				}
				else
				{
					BB = Union(BB, iter->BBox);
					nb += 1;
				}

			}
			if (na == 0 || nb == 0)
			{
				continue;
			}
			cost = 0.125f + (na * BA.surfaceArea() + nb * BB.surfaceArea()) / p->BoundingBox.surfaceArea();
			if (cost < min_cost)
			{
				target_split_point = split_point;
				min_cost = cost;
			}
		}

	}
	else if (axis == 1)
	{
		cost = 0.0f;
		//沿x轴的划分
		width = p->BoundingBox.Diagonal().y;
		for (i = 0; i < split_size; i++)
		{
			split_point = p->BoundingBox.pMin.y + i * width / split_size;
			//分成x轴向<=split_point和>split_point的元素
			auto parti_pos = std::partition(primitives.begin() + start, primitives.begin() + end, [split_point](const Primitive& prim) {return prim.BBox.centroid().y <= split_point; });
			na = 0;
			nb = 0;
			Bounds3f BA = primitives[start].BBox;
			na += 1;
			for (auto iter = primitives.begin() + start; iter != parti_pos; iter++)
			{
				if (iter == primitives.begin() + start)
				{
					continue;
				}
				else
				{
					BA = Union(BA, iter->BBox);
					na += 1;
				}

			}
			Bounds3f BB = parti_pos->BBox;
			nb += 1;
			for (auto iter = parti_pos; iter != primitives.begin() + end; iter++)
			{
				if (iter == parti_pos)
				{
					continue;
				}
				else
				{
					BB = Union(BB, iter->BBox);
					nb += 1;
				}

			}
			if (na == 0 || nb == 0)
			{
				continue;
			}
			cost = 0.125f + (na * BA.surfaceArea() + nb * BB.surfaceArea()) / p->BoundingBox.surfaceArea();
			if (cost < min_cost)
			{
				target_split_point = split_point;
				min_cost = cost;
			}
		}
	}
	else
	{
		cost = 0.0f;
		//沿x轴的划分
		width = p->BoundingBox.Diagonal().z;
		for (i = 0; i < split_size; i++)
		{
			split_point = p->BoundingBox.pMin.z + i * width / split_size;
			//分成x轴向<=split_point和>split_point的元素
			auto parti_pos = std::partition(primitives.begin() + start, primitives.begin() + end, [split_point](const Primitive& prim) {return prim.BBox.centroid().z <= split_point; });
			na = 0;
			nb = 0;
			Bounds3f BA = primitives[start].BBox;
			na += 1;
			for (auto iter = primitives.begin() + start; iter != parti_pos; iter++)
			{
				if (iter == primitives.begin() + start)
				{
					continue;
				}
				else
				{
					BA = Union(BA, iter->BBox);
					na += 1;
				}

			}
			Bounds3f BB = parti_pos->BBox;
			nb += 1;
			for (auto iter = parti_pos; iter != primitives.begin() + end; iter++)
			{
				if (iter == parti_pos)
				{
					continue;
				}
				else
				{
					BB = Union(BB, iter->BBox);
					nb += 1;
				}

			}
			if (na == 0 || nb == 0)
			{
				continue;
			}
			cost = 0.125f + (na * BA.surfaceArea() + nb * BB.surfaceArea()) / p->BoundingBox.surfaceArea();
			if (cost < min_cost)
			{
				target_split_point = split_point;
				min_cost = cost;
			}
		}
	}

	//以上代码找到了合适的划分点, 现在可以开始划分
	if (axis == 0)
	{
		auto parti_pos = std::partition(primitives.begin() + start, primitives.begin() + end, [target_split_point](const Primitive& prim) {return prim.BBox.centroid().x <= target_split_point; });
		mid = parti_pos - primitives.begin();
	}
	else if (axis == 1)
	{
		auto parti_pos = std::partition(primitives.begin() + start, primitives.begin() + end, [target_split_point](const Primitive& prim) {return prim.BBox.centroid().y <= target_split_point; });
		mid = parti_pos - primitives.begin();
	}
	else
	{
		auto parti_pos = std::partition(primitives.begin() + start, primitives.begin() + end, [target_split_point](const Primitive& prim) {return prim.BBox.centroid().z <= target_split_point; });
		mid = parti_pos - primitives.begin();
	}


	//如果没怎么分
	if (mid <= start || mid >= end)
	{
		mid = (start + end) / 2;
	}
	p->left = BuildTree(primitives, start, mid, total_num);
	p->right = BuildTree(primitives, mid, end, total_num);
	p->axis = axis;
	return p;

};

BVHNode* BuildSAH(std::vector<Primitive>& primitives, int num_primitives, int* total_num)
{
	BVHNode* node = BuildTree(primitives, 0, num_primitives, total_num);

	return node;

};

int VisitNode(BVHNode* node, std::vector<BVHInfo>& info, int* pos)
{
	int curr_pos = (*pos);
	int right_pos=curr_pos;
	(*pos) += 1;
	if (node->left != nullptr)
	{
		VisitNode(node->left, info, pos);
	}
	if (node->right != nullptr)
	{
		right_pos=VisitNode(node->right, info, pos);
	}
	info[curr_pos] = BVHInfo(node->tri_id, right_pos - curr_pos, node->is_leaf, node->BoundingBox);
	return curr_pos;
};

std::vector<BVHInfo>  DFSCampact(BVHNode* node, int total_num)
{
	int pos = 0;
	std::vector<BVHInfo> info(total_num);
	VisitNode(node, info, &pos);
	return info;

};


__global__ void constructTriangles(Primitive* primitives,const int* triangles, const float* vertices,const int num_triangles)
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


	primitives[idx].BBox=Union(Union(Bounds3(primitives[idx].points[0]),primitives[idx].points[1]),primitives[idx].points[2]);
	primitives[idx].idx=idx;

}

__global__ void WriteBVH(BVHInfo* info,const int num_nodes, int* offset,float* pMin,float* pMax,int* triangle_id)
{
	int idx=cg::this_grid().thread_rank();
	if(idx>=num_nodes)
	{
		return;
	}

	offset[idx]=info[idx].offset;
	pMin[3*idx]=info[idx].BBox.pMin.x;
	pMin[3*idx+1]=info[idx].BBox.pMin.y;
	pMin[3*idx+2]=info[idx].BBox.pMin.z;

	pMax[3*idx]=info[idx].BBox.pMax.x;
	pMax[3*idx+1]=info[idx].BBox.pMax.y;
	pMax[3*idx+2]=info[idx].BBox.pMax.z;

	triangle_id[idx]=info[idx].triangle_id;

}


int ComputeNodes(std::function<char* (size_t)> SceneInfo,
	const int* triangles,
	const float* vertices,
	const int num_triangles)
{
	//计算BVH不需要为材质分配空间
	int num_materials=1;
	size_t chunk_size=PrecomputeSceneBias(num_triangles,num_materials);
	char* chunk_ptr=SceneInfo(chunk_size);
	Scene scene=Scene::ALLOC(chunk_ptr, num_triangles,num_materials);
	
	cudaError_t err;

	//step1 给三角形赋值
	constructTriangles<<<(num_triangles+255)/256,256>>>(scene.primitives,triangles, vertices,num_triangles);
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("片元初始化失败\n");
	}

	std::vector<Primitive> host_primitive(num_triangles);
	cudaMemcpy(host_primitive.data(),scene.primitives,num_triangles*sizeof(Primitive),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("数据拷贝失败\n");
	}

	int num_nodes=0;
	BVHNode* node=BuildSAH( host_primitive, num_triangles,&num_nodes);
	return num_nodes;
}

void ContructBvhKernel(std::function<char* (size_t)> SceneInfo,
	std::function<char* (size_t)> BBoxInfo,
	const int* triangles,
	const float* vertices,
	int* offset,
	float* pMin,
	float* pMax,
	int* triangle_id,
	const int num_triangles)
{
	
	//计算BVH不需要为材质分配空间
	int num_materials=1;
	size_t chunk_size=PrecomputeSceneBias(num_triangles,num_materials);
	char* chunk_ptr=SceneInfo(chunk_size);
	Scene scene=Scene::ALLOC(chunk_ptr, num_triangles,num_materials);
	
	cudaError_t err;

	//step1 给三角形赋值
	constructTriangles<<<(num_triangles+255)/256,256>>>(scene.primitives,triangles, vertices,num_triangles);
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("片元初始化失败\n");
	}

	std::vector<Primitive> host_primitive(num_triangles);
	cudaMemcpy(host_primitive.data(),scene.primitives,num_triangles*sizeof(Primitive),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("数据拷贝失败\n");
	}

	int num_nodes=0;
	BVHNode* node=BuildSAH( host_primitive, num_triangles,&num_nodes);
	
	std::vector<BVHInfo> Host_BVHInfo=DFSCampact(node,num_nodes);

	size_t bvh_chunk_size=PrecomputeBVHBufferBias(num_nodes);
	char* bvh_chunk_ptr=BBoxInfo(bvh_chunk_size);
	BVHBuffer b=BVHBuffer::ALLOC(bvh_chunk_ptr, num_nodes);

	cudaMemcpy(b.bvhs,Host_BVHInfo.data(),num_nodes*sizeof(BVHInfo),cudaMemcpyHostToDevice );
	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("SAH主机到设备数据拷贝失败\n");
	}


	WriteBVH<<<(num_nodes+255)/256,256>>>(b.bvhs,num_nodes, offset,pMin,pMax,triangle_id);

	err=cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("SAH写入torch数据失败\n");
	}
}
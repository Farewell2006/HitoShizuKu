#pragma once
#include <math_utils.h>
#include <render.h>
#include <algorithm>

struct BVHNode
{
	Bounds3f BoundingBox;
	BVHNode* left;
	BVHNode* right;
	int axis;
	int start, end;

	bool is_leaf;
	int tri_id;

	BVHNode() = default;

	BVHNode(Bounds3f b, int axis,int start,int end,bool is_leaf):BoundingBox(b),axis(axis),start(start),end(end),is_leaf(is_leaf)
	{
		left = nullptr;
		right = nullptr;
	}

};


BVHNode* BuildTree(std::vector<Primitive>& primitives, int start, int end,int* total_num);

BVHNode* BuildSAH(std::vector<Primitive>& primitives, int num_primitives,int* total_num);

struct BVHInfo
{
	int triangle_id;
	int offset;
	bool is_leaf;
	Bounds3f BBox;

	BVHInfo() = default;

	BVHInfo(int triangle_id, int offset, bool is_leaf, Bounds3f b) :triangle_id(triangle_id), offset(offset), is_leaf(is_leaf), BBox(b)
	{

	}

};

struct BVHBuffer
{
	BVHInfo* bvhs;
	int num_nodes;

	static BVHBuffer ALLOC(char*& address, size_t num_nodes);
};

static size_t PrecomputeBVHBufferBias(size_t num_nodes)
{
	char* size = nullptr;
	BVHBuffer::ALLOC(size,num_nodes);
	return ((size_t)size);
};


int VisitNode(BVHNode* node, std::vector<BVHInfo>& info, int* pos);

std::vector<BVHInfo>  DFSCampact(BVHNode* node, int total_num);



int ComputeNodes(std::function<char* (size_t)> SceneInfo, 
	const int* triangles,
	const float* vertices,
	const int num_triangles);

void ContructBvhKernel(std::function<char* (size_t)> SceneInfo,
	std::function<char* (size_t)> BBoxInfo,
	const int* triangles,
	const float* vertices,
	int* offset,
	float* pMin,
	float* pMax,
	int* triangle_id,
	const int num_triangles);



#include <rasterizer.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda.h>
#include <fstream>
#include <cub/cub.cuh>

namespace cg=cooperative_groups;


__global__ BackShade()

void RasterizationBWD(
						int num_vertices,
						int num_triangles,
						int num_uvs,
						const float* background,
						const int* triangles,
						const float* vertices,
						const float* uvs,
						const int* uv_index,
						const float* texture,
						const float* proj_matrix,
						const float tan_fovx,
						const float tan_fovy,
						const int H,
						const int W,
						const int texture_H,
						const int texture_W,
						const int* pix_contributor,
						const float* dO,
						float* dscreen_coor,
						float* dV,
						float* dtexture,
						)
{
	//backward的实现要简单的多 因为我们已经在forward中记录了对每个像素有贡献的三角形索引(不超过每个像素采样的数目)
	
	dim3 block_size(BLOCK_X,BLOCK_Y);
	dim3 grid_size((W+BLOCK_X-1)/BLOCK_X,(H+BLOCK_Y-1)/BLOCK_Y);
	
}
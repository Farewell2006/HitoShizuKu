#include <torch/extension.h>
#include <cstdio>
#include <tuple>


//光栅化的forward函数
std::tuple<torch::Tensor, torch::Tensor>
RasterizationForward(const torch::Tensor& background,
	const torch::Tensor& triangles,
	const torch::Tensor& vertices,
	const torch::Tensor& uvs,
	const torch::Tensor& uv_index,
	const torch::Tensor& texture,
	const torch::Tensor& f_mtl,
	const torch::Tensor& view_matrix,
	const torch::Tensor& proj_matrix,
	const float tan_fovx,
	const float tan_fovy,
	const int H,
	const int W);

//BVH构建函数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor>
ConstructBVH(const torch::Tensor& triangles,
	const torch::Tensor& vertices);

//路径追踪的forwad函数
std::tuple<torch::Tensor, torch::Tensor,torch::Tensor>
PathTarceForward(
	const torch::Tensor& texture_id,
	const torch::Tensor& triangles,
	const torch::Tensor& vertices,
	const torch::Tensor& uvs,
	const torch::Tensor& uv_index,
	const torch::Tensor& texture,
	const torch::Tensor& f_mtl,
	const torch::Tensor& cam_to_world,
	const torch::Tensor& offset,
	const torch::Tensor& pMin,
	const torch::Tensor& pMax,
	const torch::Tensor& triangle_id,
	const torch::Tensor& ray_bias,
	const int H,
	const int W,
	const float tan_fovy,
	const float tan_fovx,
	const int num_spp);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PathTraceBackward(const torch::Tensor& triangles,
	const torch::Tensor& vertices,
	const torch::Tensor& uvs,
	const torch::Tensor& uv_index,
	const torch::Tensor& texture,
	const torch::Tensor& cam_to_world,
	const torch::Tensor& ray_bias,
	const torch::Tensor& intersected_tris,
	const torch::Tensor& res_text_id,
	const int H,
	const int W,
	const float tan_fovy,
	const float tan_fovx,
	const int num_spp,
	const torch::Tensor& dO,
	torch::Tensor& d_vertices,
	torch::Tensor& d_uv,
	torch::Tensor& d_texture);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("RasterizationForward", &RasterizationForward, "CUDA加速的前向光栅化");
	m.def("PathTarceForward", &PathTarceForward, "CUDA加速的前向路径追踪");
	m.def("ConstructBVH", &ConstructBVH, "SAH构建函数");
	m.def("PathTraceBackward", &PathTraceBackward, "CUDA加速的反向路径追踪");
}
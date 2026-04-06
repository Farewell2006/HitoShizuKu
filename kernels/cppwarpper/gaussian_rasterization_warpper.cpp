#include <torch/extension.h>
#include <cstdio>
#include <tuple> 

// 3DGS的forward函数, 参数基本和官方实现相同, 这里默认3D协方差是在CUDA中计算的.
std::tuple<int, torch::Tensor, torch::Tensor,torch::Tensor, torch::Tensor>
GaussianRasterizationForward(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& view_matrix,
	const torch::Tensor& proj_matrix,
	const float tan_fovx,
	const float tan_fovy,
	const int H,
	const int W);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizationBackward(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& view_matrix,
	const torch::Tensor& proj_matrix,
	const float tan_fovx,
	const float tan_fovy,
	const int H,
	const int W,
	const torch::Tensor& dO,
	const torch::Tensor geometryBuffer,
	const torch::Tensor binningBuffer,
	const torch::Tensor ImgBuffer,
	int num_rendered);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("GaussianRasterizationForward", &GaussianRasterizationForward, "CUDA加速的Gaussian泼溅光栅化forward");
	m.def("GaussianRasterizationBackward", &GaussianRasterizationBackward, "CUDA加速的Gaussian泼溅光栅化backward");
}


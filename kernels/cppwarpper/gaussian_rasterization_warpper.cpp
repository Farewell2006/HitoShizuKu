#include <torch/extension.h>
#include <cstdio>
#include <tuple> 

// 3DGS的forward函数, 参数基本和官方实现相同, 这里默认3D协方差是在CUDA中计算的.
std::tuple<int, torch::Tensor, torch::Tensor,torch::Tensor>
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
	const int W,
	const torch::Tensor& spherical_harmonics,
	const int degree,
	const torch::Tensor& camera_position);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("GaussianRasterizationForward", &GaussianRasterizationForward, "CUDA加速的Gaussian泼溅光栅化");
}


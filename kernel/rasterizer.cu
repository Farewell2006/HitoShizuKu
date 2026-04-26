#include <torch/extension.h>
#include <cstdio>
#include <rasterizer.h>

inline std::function<char*(size_t N)> CaptureAndResize(torch::Tensor& x)
{
	//按引用捕获tensor x, 以后的修改都会直接修改x
	auto lambda=[&x](size_t N)
	{
		x.resize_({(long long)N});
		return reinterpret_cast<char*>(x.contiguous().data_ptr());
	};
	return lambda;
}

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
	const int W)
{
	int num_triangles=triangles.size(0);
	int num_vertices=vertices.size(0);
	int num_uvs=uvs.size(0);
	int texture_H=texture.size(1);
	int texture_W=texture.size(2);

	auto float_ops=vertices.options().dtype(torch::kFloat32);
	torch::Tensor res=torch::full({H,W,3},0.0,float_ops);
	
	torch::Device device(torch::kCUDA);

	torch::TensorOptions options(torch::kByte);

	torch::Tensor geometryBuffer=torch::empty({0},options.device(device));
	torch::Tensor IntermediateBuffer=torch::empty({0},options.device(device));
	torch::Tensor TriangleBuffer=torch::empty({0},options.device(device));
	torch::Tensor ImageBuffer=torch::empty({0},options.device(device));

	std::function<char* (size_t)> geometryInfo=CaptureAndResize(geometryBuffer);
	std::function<char* (size_t)> intermediateInfo=CaptureAndResize(IntermediateBuffer);
	std::function<char* (size_t)> triangleInfo=CaptureAndResize(TriangleBuffer);
	std::function<char* (size_t)> imageInfo=CaptureAndResize(ImageBuffer);

	RasterizationFWD(
						geometryInfo,
						intermediateInfo,
						triangleInfo,
						imageInfo,
						num_vertices,
						num_triangles,
						num_uvs,
						background.contiguous().data<float>(),
						triangles.contiguous().data<int>(),
						vertices.contiguous().data<float>(),
						uvs.contiguous().data<float>(),
						uv_index.contiguous().data<int>(),
						texture.contiguous().data<float>(),
						f_mtl.contiguous().data<int>(),
						proj_matrix.contiguous().data<float>(),
						tan_fovx,
						tan_fovy,
						H,
						W,
						texture_H,
						texture_W,
						res.contiguous().data<float>());


	fflush(stdout);

	return std::make_tuple(res,res);

}
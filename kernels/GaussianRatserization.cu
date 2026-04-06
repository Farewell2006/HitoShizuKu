#include <torch/extension.h>
#include <cstdio>
#include <3dgs_forward.h>
#include <3dgs_structure.h>
#include <3dgs_backward.h>



std::function<char*(size_t N)> resizeFunctional(torch::Tensor& x)
{
	//按引用捕获tensor x, 以后的修改都会直接修改x
	auto lambda=[&x](size_t N)
	{
		x.resize_({(long long)N});
		return reinterpret_cast<char*>(x.contiguous().data_ptr());
	};
	return lambda;
}


std::tuple<int, torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
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
	const int W)
{
	
	
	int num_points=means3D.size(0);
	//获取means3D的设置，自动对齐输入设备.
	auto float_opts=means3D.options().dtype(torch::kFloat32);
	//一些渲染信息, res代表渲染结果 radius代表2D屏幕上整数值的Gaussian球体半径
	torch::Tensor res=torch::full({3,H,W},0.0,float_opts);
	
	
	
	//一块GPU
	torch::Device device(torch::kCUDA);

	torch::TensorOptions options(torch::kByte);
	//初始化一些渲染过程中需要的信息, 可以在3dgs_structure.h中找到
	torch::Tensor geometryBuffer=torch::empty({0},options.device(device));
	torch::Tensor binningBuffer=torch::empty({0},options.device(device));
	torch::Tensor ImgBuffer=torch::empty({0},options.device(device));

	//刚才初始化geometryBuffer的时候只有一个元素, 现在geometryFunction是一个函数, 给它一个参数N就能把geometryBuffer修改成大小为N的tensor
	std::function<char*(size_t)> geometryFunction=resizeFunctional(geometryBuffer);
	std::function<char*(size_t)> binnFunction=resizeFunctional(binningBuffer);
	std::function<char*(size_t)> ImgFunction=resizeFunctional(ImgBuffer);

	int rendered=0;
	

	
	int M=0;
	rendered=gsForward(
						geometryFunction,
						binnFunction,
						ImgFunction,
						background.contiguous().data<float>(),
						num_points,
						means3D.contiguous().data<float>(),
						colors.contiguous().data<float>(),
						scales.contiguous().data_ptr<float>(),
						scale_modifier,
						rotations.contiguous().data<float>(),
						opacity.contiguous().data<float>(),
						view_matrix.contiguous().data<float>(),
						proj_matrix.contiguous().data<float>(),
						H,W,
						tan_fovx,
						tan_fovy,
						res.contiguous().data<float>());
	fflush(stdout);

	return std::make_tuple(rendered,res,geometryBuffer,binningBuffer,ImgBuffer);
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
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
								int num_rendered)
{
	const int P=means3D.size(0);

	torch::Tensor dL_dmeans2D=torch::zeros({P,2},means3D.options());
	torch::Tensor dL_dmeans3D=torch::zeros({P,3},means3D.options());

	torch::Tensor dL_dcolors=torch::zeros({P,3},means3D.options());
	torch::Tensor dL_dcov2Dinv=torch::zeros({P,4},means3D.options());

	torch::Tensor dL_dopacity=torch::zeros({P,1},means3D.options());
	torch::Tensor dL_dcov3D=torch::zeros({P,6},means3D.options());
	torch::Tensor dL_dscale=torch::zeros({P,3},means3D.options());
	torch::Tensor dL_dq=torch::zeros({P,4},means3D.options());

	gsBackward(
				reinterpret_cast<char*>(geometryBuffer.contiguous().data_ptr()),
				reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
				reinterpret_cast<char*>(ImgBuffer.contiguous().data_ptr()),
				H,W,
				P,
				num_rendered,
				background.contiguous().data<float>(),
				means3D.contiguous().data<float>(),
				colors.contiguous().data<float>(),
				scales.contiguous().data_ptr<float>(),
				scale_modifier,
				rotations.contiguous().data<float>(),
				view_matrix.contiguous().data<float>(),
				proj_matrix.contiguous().data<float>(),
				tan_fovx,
				tan_fovy,
				dO.contiguous().data<float>(),
				dL_dmeans2D.contiguous().data<float>(),
				dL_dcov2Dinv.contiguous().data<float>(),
				dL_dopacity.contiguous().data<float>(),
				dL_dcolors.contiguous().data<float>(),
				dL_dmeans3D.contiguous().data<float>(),
				dL_dcov3D.contiguous().data<float>(),
				dL_dscale.contiguous().data<float>(),
				dL_dq.contiguous().data<float>());
	fflush(stdout);

	return std::make_tuple(dL_dmeans3D,dL_dcolors, dL_dscale, dL_dopacity,dL_dq);
}
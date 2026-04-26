#include <torch/extension.h>
#include <cstdio>
#include <render.h>
#include <sah.h>

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



std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
ConstructBVH(const torch::Tensor& triangles,
			const torch::Tensor& vertices)
{
	int num_triangles=triangles.size(0);
	int num_vertices=vertices.size(0);
	auto float_ops=vertices.options().dtype(torch::kFloat32);
	auto int_ops=triangles.options().dtype(torch::kInt32);

	torch::Device device(torch::kCUDA);

	torch::TensorOptions options(torch::kByte);
	torch::Tensor SceneBuffer=torch::empty({0},options.device(device));
	torch::Tensor BVHBuffer=torch::empty({0},options.device(device));

	

	std::function<char* (size_t)> SceneInfo=CaptureAndResize(SceneBuffer);
	std::function<char* (size_t)> BBoxInfo=CaptureAndResize(BVHBuffer);

	int num_nodes=ComputeNodes(SceneInfo, 
	triangles.contiguous().data<int>(),
	vertices.contiguous().data<float>(),
	num_triangles);


	torch::Tensor offset=torch::full({num_nodes},0,int_ops);
	torch::Tensor pMin=torch::full({num_nodes,3},0.0,float_ops);
	torch::Tensor pMax=torch::full({num_nodes,3},0.0,float_ops);
	torch::Tensor triangle_id=torch::full({num_nodes},-1,int_ops);

	ContructBvhKernel(SceneInfo,
		BBoxInfo,
		triangles.contiguous().data<int>(),
		vertices.contiguous().data<float>(),
		offset.contiguous().data<int>(),
		pMin.contiguous().data<float>(),
		pMax.contiguous().data<float>(),
		triangle_id.contiguous().data<int>(),
		num_triangles);

	return std::make_tuple(offset,pMin,pMax,triangle_id);
}

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
	const int num_spp)
{
	int num_triangles=triangles.size(0);
	int num_vertices=vertices.size(0);
	int num_uvs=uvs.size(0);
	int num_materials=texture_id.size(0);
	int num_nodes=pMin.size(0);
	int texture_H=texture.size(1);
	int texture_W=texture.size(2);

	auto float_ops=vertices.options().dtype(torch::kFloat32);
	torch::Tensor res=torch::full({H,W,3},0.0,float_ops);
	
	torch::Device device(torch::kCUDA);

	torch::TensorOptions options(torch::kByte);

	torch::Tensor SceneBuffer=torch::empty({0},options.device(device));
	torch::Tensor RayBuffer=torch::empty({0},options.device(device));
	torch::Tensor BVHBuffer=torch::empty({0},options.device(device));
	torch::Tensor InteractBuffer=torch::empty({0},options.device(device));

	std::function<char* (size_t)> SceneInfo=CaptureAndResize(SceneBuffer);
	std::function<char* (size_t)> RayInfo=CaptureAndResize(RayBuffer);
	std::function<char* (size_t)> BVHInfo=CaptureAndResize(BVHBuffer);
	std::function<char* (size_t)> InteractionInfo=CaptureAndResize(InteractBuffer);

	//一些方便进行反向传播的计算信息
	auto int_ops=triangles.options().dtype(torch::kInt32);
	torch::Tensor intersected_tris=torch::full({H,W,num_spp},-1,int_ops);
	torch::Tensor res_text_id=torch::full({H,W,num_spp},-1,int_ops);

	PathTraceFWD(SceneInfo,
					RayInfo,
					BVHInfo,
					InteractionInfo,
					triangles.contiguous().data<int>(),
					vertices.contiguous().data<float>(),
					uvs.contiguous().data<float>(),
					uv_index.contiguous().data<int>(),
					texture.contiguous().data<float>(),
					f_mtl.contiguous().data<int>(),
					cam_to_world.contiguous().data<float>(),
					offset.contiguous().data<int>(),
					pMin.contiguous().data<float>(),
					pMax.contiguous().data<float>(),
					triangle_id.contiguous().data<int>(),
					ray_bias.contiguous().data<float>(),
					tan_fovy,
					tan_fovx,
					H,
					W,
					texture_id.contiguous().data<int>(),
					num_triangles,
					num_uvs,
					num_vertices,
					num_materials,
					num_nodes,
					texture_H,
					texture_W,
					num_spp,
					res.contiguous().data<float>(),
					intersected_tris.contiguous().data<int>(),
					res_text_id.contiguous().data<int>());

	fflush(stdout);
	return std::make_tuple(res,intersected_tris,res_text_id);
	
}


std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>
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
	torch::Tensor& d_texture)
{
	auto float_ops=vertices.options().dtype(torch::kFloat32);
	torch::Tensor d_b0=torch::full({H,W},0.0,float_ops);
	torch::Tensor d_b1=torch::full({H,W},0.0,float_ops);
	int texture_H=texture.size(1);
	int texture_W=texture.size(2);


	PathTraceBWD(
					intersected_tris.contiguous().data<int>(),
					res_text_id.contiguous().data<int>(),
					triangles.contiguous().data<int>(),
					vertices.contiguous().data<float>(),
					uvs.contiguous().data<float>(),
					uv_index.contiguous().data<int>(),
					texture.contiguous().data<float>(),
					cam_to_world.contiguous().data<float>(),
					ray_bias.contiguous().data<float>(),
					H,
					W,
					texture_H,
					texture_W,
					tan_fovy,
					tan_fovx,
					num_spp,
					dO.contiguous().data<float>(),
					d_b0.contiguous().data<float>(),
					d_b1.contiguous().data<float>(),
					d_vertices.contiguous().data<float>(),
					d_uv.contiguous().data<float>(),
					d_texture.contiguous().data<float>());

	fflush(stdout);
	return std::make_tuple( d_vertices,d_uv,d_texture);
}
#pragma once

void gsBackward(
	char* geometryBuffer,
	char* binningBuffer,
	char* imageBuffer,
	int H, int W,
	int num_points,
	int num_rendered,
	float* background,
	float* means3D,
	float* colors,
	float* scales,
	const float scale_modifier,
	float* quant_number,
	float* view_matrix,
	float* proj_matrix,
	float tan_fovx,
	float tan_fovy,
	const float* dL_dres,
	float* dL_dmeans2D,
	float* dL_dcov2Dinv,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmeans3D,
	float* dL_dcov3D,
	float* dL_dscale,
	float* dL_dq);


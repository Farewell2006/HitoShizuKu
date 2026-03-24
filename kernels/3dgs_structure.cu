#include <3dgs_structure.h>
#include <cuda.h>
#include <cub/cub.cuh>


Geometry Geometry::fromChunk(char*& chunk,size_t N)
{
	Geometry g;
	manual_allocate(chunk,g.means2D,N,128);
	manual_allocate(chunk,g.colors,3*N,128);
	manual_allocate(chunk,g.opacity,N,128);
	manual_allocate(chunk,g.depth,N,128);
	manual_allocate(chunk,g.cov3D,6*N,128);
	manual_allocate(chunk,g.invcoc2D_opacity,N,128);
	manual_allocate(chunk,g.radius,N,128);
	manual_allocate(chunk,g.scanned_points,N,128);
	manual_allocate(chunk,g.insected_tiles,N,128);

	cub::DeviceScan::InclusiveSum(nullptr,g.scan_size,g.insected_tiles,g.insected_tiles,N);
	manual_allocate(chunk,g.scan_address,g.scan_size,128);
	return g;
}

Binning Binning::fromChunk(char*& chunk,size_t N)
{
	Binning b;
	
	manual_allocate(chunk,b.keys_unsorted, N,128);
	manual_allocate(chunk,b.keys_sorted, N,128);
	manual_allocate(chunk,b.pointsIdx_unsorted, N,128);
	manual_allocate(chunk,b.pointIdx_sorted, N,128);

	cub::DeviceRadixSort::SortPairs(nullptr,b.sort_size,b.keys_unsorted,b.keys_sorted,b.pointsIdx_unsorted,b.pointIdx_sorted,N);
	manual_allocate(chunk,b.list_sorting_space, b.sort_size,128);
	return b;
};

Image Image::fromChunk(char*& chunk,size_t N)
{
	Image img;
	manual_allocate(chunk,img.ranges, N,128);
	manual_allocate(chunk,img.num_contributor, N,128);
	manual_allocate(chunk,img.alpha, N,128);
	return img;
};
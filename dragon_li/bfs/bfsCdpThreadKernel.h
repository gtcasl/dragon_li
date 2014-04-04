#pragma once

template< typename Settings >
__global__ void bfsCdpThreadExpandKernel(
	typename Settings::SizeType rowOffset,
	typename Settings::SizeType rowLength,
	typename Settings::VertexId * devColumnIndices,
	typename Settings::VertexId * devFrontierExpand) {

}

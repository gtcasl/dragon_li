#pragma once

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace bfs {

template< typename Settings >
__global__ void bfsCdpThreadExpandKernel(
	typename Settings::SizeType rowOffset,
	typename Settings::SizeType rowLength,
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::VertexIdType * devFrontierExpand,
	typename Settings::SizeType globalOffset,
	typename Settings::SizeType localOffset) {
		
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::VertexIdType VertexIdType;

	SizeType columnId = threadIdx.x + blockIdx.x * blockDim.x;
	if(columnId < rowLength) {

		VertexIdType expandedVertex = devColumnIndices[rowOffset + columnId];
		SizeType outputOffset = globalOffset + localOffset + columnId;
		devFrontierExpand[outputOffset]	= expandedVertex;

		reportDevice("CDP %d.%d: vertex %d, outputoffset %d", 
			blockIdx.x, threadIdx.x, expandedVertex, outputOffset);
	}
}

}
}

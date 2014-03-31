#pragma once

#include <dragon_li/util/primitive.h>
#include <dragon_li/util/ctaOutputAssignment.h>

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsRegDevice {
	
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;

public:
	class CtaWorkAssignment {
	public:
		SizeType totalWorkSize;
		SizeType workOffset;
		SizeType workSize;

		__device__ CtaWorkAssignment(SizeType _totalWorkSize) : 
			totalWorkSize(_totalWorkSize),
			workOffset(-1),
			workSize(0) {}

		__device__ void getCtaWorkAssignment() {
			SizeType totalThreads = THREADS * CTAS;
			if(workOffset == -1) { //first time
				workOffset = min(blockIdx.x * THREADS, totalWorkSize);
			}
			else {
				workOffset = min(workOffset + totalThreads, totalWorkSize);
			}
			workSize = min(THREADS, totalWorkSize - workOffset);
		}
	};

	static __device__ void bfsRegCtaSearch(
		CtaWorkAssignment &ctaWorkAssignment,
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		VertexIdType * devFrontierIn,
		VertexIdType * devFrontierOut,
		CtaOutputAssignment & ctaOutputAssignment) {


		VertexIdType vertexId = -1;
		SizeType rowOffset = -1;
		SizeType nextRowOffset = -1;
		SizeType rowLength = 0;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			vertexId = devFrontierIn[ctaWorkAssignment.workOffset + threadIdx.x];
			rowOffset = devRowOffsets[vertexId];
			nextRowOffset = devRowOffsets[vertexId + 1];
			rowLength = nextRowOffset - rowOffset;
		}

		SizeType totalOutputCount;
		SizeType localOffset; //output offset within cta
		localOffset = dragon_li::util::prefixSumCta<THREADS, SizeType>(rowLength, 
				totalOutputCount);

		__shared__ SizeType globalOffset;

		if(threadIdx.x == 0 && totalOutputCount > 0) {
			globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalOutputCount);
			printf("cta %d, global off = %d\n", blockIdx.x, globalOffset);
		}

		__syncthreads();

		for(SizeType columnId = 0; columnId < rowLength; columnId++) {
			VertexIdType neighborVertexId = devColumnIndices[rowOffset + columnId];
			devFrontierOut[globalOffset + localOffset + columnId] = neighborVertexId;
			printf("%d.%d: neighbor %d, out index %d\n", blockIdx.x, threadIdx.x, neighborVertexId, globalOffset + localOffset + columnId);
		}
		
	}


	static __device__ void bfsRegSearchKernel(
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		VertexIdType * devFrontierIn,
		VertexIdType * devFrontierOut,
		SizeType maxFrontierSize,
		SizeType * devFrontierSize,
		CtaOutputAssignment & ctaOutputAssignment) {

		SizeType frontierSize = *devFrontierSize;

		CtaWorkAssignment ctaWorkAssignment(frontierSize);


		while(ctaWorkAssignment.workOffset < frontierSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			bfsRegCtaSearch(
				ctaWorkAssignment,
				devColumnIndices,
				devRowOffsets,
				devFrontierIn,
				devFrontierOut,
				ctaOutputAssignment);
		}


	}
};


template< typename Settings >
__global__ void bfsRegSearchKernel(
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::SizeType * devRowOffsets,
	typename Settings::VertexIdType * devFrontierIn,
	typename Settings::VertexIdType * devFrontierOut,
	typename Settings::SizeType maxFrontierSize,
	typename Settings::SizeType * devFrontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	BfsRegDevice< Settings >::bfsRegSearchKernel(
					devColumnIndices,
					devRowOffsets,
					devFrontierIn,
					devFrontierOut,
					maxFrontierSize,
					devFrontierSize,
					ctaOutputAssignment);

}

}
}

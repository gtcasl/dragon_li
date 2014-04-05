#pragma once

#include <dragon_li/bfs/bfsRegDevice.h>
#include <dragon_li/bfs/bfsCdpThread.h>

#undef REPORT_BASE
#define REPORT_BASE 1

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsCdpDevice {

	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::MaskType MaskType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename BfsRegDevice< Settings >::CtaWorkAssignment CtaWorkAssignment;

public:

	static __device__ void bfsCdpCtaExpand(
		CtaWorkAssignment &ctaWorkAssignment,
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		SizeType * devSearchDistance,
		VertexIdType * devFrontierContract,
		VertexIdType * devFrontierExpand,
		SizeType maxFrontierSize,
		CtaOutputAssignment & ctaOutputAssignment,
		SizeType iteration) {


		VertexIdType vertexId = -1;
		SizeType rowOffset = -1;
		SizeType nextRowOffset = -1;
		SizeType rowLength = 0;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			vertexId = devFrontierContract[ctaWorkAssignment.workOffset + threadIdx.x];

			SizeType searchDistance = devSearchDistance[vertexId];
			if(searchDistance == -1)
				devSearchDistance[vertexId] = iteration;
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
		}

		__syncthreads();

		if(ctaOutputAssignment.getGlobalSize() > maxFrontierSize) //overflow
			return;

		if(rowLength >= Settings::CDP_THRESHOLD) { //call cdp kernel

			SizeType CDP_THREADS = Settings::CDP_THREADS;
			SizeType cdpCtas = (rowLength + CDP_THREADS - 1) >> Settings::CDP_THREADS_BITS;

			bfsCdpThreadExpandKernel<Settings>
				<<< cdpCtas, CDP_THREADS >>> (
					rowOffset,
					rowLength,
					devColumnIndices,
					devFrontierExpand);

			cudaError_t retVal = cudaGetLastError();
			if(retVal)

			rowLength = 0;
		}


		for(SizeType columnId = 0; columnId < rowLength; columnId++) {
			VertexIdType neighborVertexId = devColumnIndices[rowOffset + columnId];
			devFrontierExpand[globalOffset + localOffset + columnId] = neighborVertexId;
			reportDevice("%d.%d, neighborid %d, outputoffset %d\n", blockIdx.x, threadIdx.x, neighborVertexId, globalOffset + localOffset + columnId);
		}
		
	}


	static __device__ void bfsCdpExpandKernel(
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		SizeType * devSearchDistance,
		VertexIdType * devFrontierContract,
		VertexIdType * devFrontierExpand,
		SizeType maxFrontierSize,
		SizeType frontierSize,
		CtaOutputAssignment & ctaOutputAssignment,
		SizeType iteration) {

		CtaWorkAssignment ctaWorkAssignment(frontierSize);


		while(ctaWorkAssignment.workOffset < frontierSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			bfsCdpCtaExpand(
				ctaWorkAssignment,
				devColumnIndices,
				devRowOffsets,
				devSearchDistance,
				devFrontierContract,
				devFrontierExpand,
				maxFrontierSize,
				ctaOutputAssignment,
				iteration);
		}


	}

};


template< typename Settings >
__global__ void bfsCdpExpandKernel(
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::SizeType * devRowOffsets,
	typename Settings::SizeType * devSearchDistance,
	typename Settings::VertexIdType * devFrontierContract,
	typename Settings::VertexIdType * devFrontierExpand,
	typename Settings::SizeType maxFrontierSize,
	typename Settings::SizeType frontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment,
	typename Settings::SizeType iteration) {

	BfsCdpDevice< Settings >::bfsCdpExpandKernel(
					devColumnIndices,
					devRowOffsets,
					devSearchDistance,
					devFrontierContract,
					devFrontierExpand,
					maxFrontierSize,
					frontierSize,
					ctaOutputAssignment,
					iteration);

}

}
}

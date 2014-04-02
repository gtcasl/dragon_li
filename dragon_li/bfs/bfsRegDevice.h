#pragma once

#include <dragon_li/util/primitive.h>
#include <dragon_li/util/ctaOutputAssignment.h>

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsRegDevice {
	
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::MaskType MaskType;
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

	static __device__ void bfsRegCtaExpand(
		CtaWorkAssignment &ctaWorkAssignment,
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		VertexIdType * devFrontierContract,
		VertexIdType * devFrontierExpand,
		CtaOutputAssignment & ctaOutputAssignment) {


		VertexIdType vertexId = -1;
		SizeType rowOffset = -1;
		SizeType nextRowOffset = -1;
		SizeType rowLength = 0;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			vertexId = devFrontierContract[ctaWorkAssignment.workOffset + threadIdx.x];
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

		for(SizeType columnId = 0; columnId < rowLength; columnId++) {
			VertexIdType neighborVertexId = devColumnIndices[rowOffset + columnId];
			devFrontierExpand[globalOffset + localOffset + columnId] = neighborVertexId;
			//printf("%d.%d, neighborid %d, outputoffset %d\n", blockIdx.x, threadIdx.x, neighborVertexId, globalOffset + localOffset + columnId);
		}
		
	}


	static __device__ void bfsRegCtaContract(
		CtaWorkAssignment &ctaWorkAssignment,
		MaskType * devVisitedMasks,
		VertexIdType * devOriginalFrontier,
		VertexIdType * devContractedFrontier,
		CtaOutputAssignment & ctaOutputAssignment) {

		VertexIdType vertexId = -1;

		if(threadIdx.x < ctaWorkAssignment.workSize) {

			vertexId = devOriginalFrontier[ctaWorkAssignment.workOffset + threadIdx.x];

			SizeType maskLocation = vertexId >> Settings::MASK_BITS; 

			SizeType maskBitLocation = 1 << (vertexId & Settings::MASK_MASK);

			MaskType entireMask = devVisitedMasks[maskLocation];

			if(entireMask & maskBitLocation) { //visited
				vertexId = -1;	
			}
			else { //not visited
				entireMask |= maskBitLocation;
				devVisitedMasks[maskLocation] = entireMask;
			}
		}

		SizeType validVertex = (vertexId == -1 ? 0 : 1);
		SizeType totalOutputCount;
		SizeType localOffset;
		localOffset = dragon_li::util::prefixSumCta<THREADS, SizeType>(validVertex,
				totalOutputCount);

		__shared__ SizeType globalOffset;

		if(threadIdx.x == 0 && totalOutputCount > 0) {
			globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalOutputCount);
		}

		__syncthreads();
		
		if(vertexId != -1) {
			devContractedFrontier[globalOffset + localOffset] = vertexId;
			//printf("%d.%d, vertex %d, outputoffset %d\n", blockIdx.x, threadIdx.x, vertexId, globalOffset + localOffset);
		}

	}

	static __device__ void bfsRegExpandKernel(
		VertexIdType * devColumnIndices,
		SizeType * devRowOffsets,
		VertexIdType * devFrontierContract,
		VertexIdType * devFrontierExpand,
		SizeType maxFrontierSize,
		SizeType frontierSize,
		CtaOutputAssignment & ctaOutputAssignment) {

		CtaWorkAssignment ctaWorkAssignment(frontierSize);


		while(ctaWorkAssignment.workOffset < frontierSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			bfsRegCtaExpand(
				ctaWorkAssignment,
				devColumnIndices,
				devRowOffsets,
				devFrontierContract,
				devFrontierExpand,
				ctaOutputAssignment);
		}


	}

	static __device__ void bfsRegContractKernel(
		MaskType * devVisitedMasks,
		VertexIdType * devOriginalFrontier,
		VertexIdType * devContractedFrontier,
		SizeType maxFrontierSize,
		SizeType frontierSize,
		CtaOutputAssignment & ctaOutputAssignment) {

		CtaWorkAssignment ctaWorkAssignment(frontierSize);

		while(ctaWorkAssignment.workOffset < frontierSize) {

			ctaWorkAssignment.getCtaWorkAssignment();

			bfsRegCtaContract(
				ctaWorkAssignment,
				devVisitedMasks,
				devOriginalFrontier,
				devContractedFrontier,
				ctaOutputAssignment);

		}

	}
		
};


template< typename Settings >
__global__ void bfsRegExpandKernel(
	typename Settings::VertexIdType * devColumnIndices,
	typename Settings::SizeType * devRowOffsets,
	typename Settings::VertexIdType * devFrontierContract,
	typename Settings::VertexIdType * devFrontierExpand,
	typename Settings::SizeType maxFrontierSize,
	typename Settings::SizeType frontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	BfsRegDevice< Settings >::bfsRegExpandKernel(
					devColumnIndices,
					devRowOffsets,
					devFrontierContract,
					devFrontierExpand,
					maxFrontierSize,
					frontierSize,
					ctaOutputAssignment);

}

template< typename Settings >
__global__ void bfsRegContractKernel(
	typename Settings::MaskType * devVisistedMasks,
	typename Settings::VertexIdType * devOriginalFrontier,
	typename Settings::VertexIdType * devContractedFrontier,
	typename Settings::SizeType maxFrontierSize,
	typename Settings::SizeType frontierSize,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	BfsRegDevice< Settings >::bfsRegContractKernel(
					devVisistedMasks,
					devOriginalFrontier,
					devContractedFrontier,
					maxFrontierSize,
					frontierSize,
					ctaOutputAssignment);
	
	
}

}
}

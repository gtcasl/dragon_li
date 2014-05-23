#pragma once

#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>

#include <dragon_li/amr/amrCdpThread.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace amr {

template< typename Settings >
class AmrCdpDevice {

	typedef typename Settings::DataType DataType;
	typedef typename Settings::SizeType SizeType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename dragon_li::util::CtaWorkAssignment<Settings> CtaWorkAssignment;
public:

	static __device__ void amrCdpCtaRefine(
		DataType * devGridData,
		SizeType * devGridPointer,
		SizeType maxGridDataSize,
		SizeType maxRefineLevel,
		CtaOutputAssignment ctaOutputAssignment,
		SizeType refineLevel) {

		DataType * devGridDataStart = devGridData + processGridOffset;
		SizeType * devGridPointerStart = devGridPointer + processGridOffset;
		DataType gridData;
		SizeType gridPointer;
		SizeType refineSize = 0;

		SizeType threadWorkOffset = ctaWorkAssignment.workOffset + threadIdx.x;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			gridData = devGridDataStart[threadWorkOffset];
			gridPointer = devGridPointerStart[threadWorkOffset];

			if(gridPointer == -1) { //Not processed
				devGridPointerStart[threadWorkOffset] = -2; //processed
				if(gridData >= gridRefineThreshold) {
					refineSize = GRID_REFINE_SIZE; 
				}

			}
		}

		SizeType totalRefineSize;
		SizeType localOffset; //output offset within cta
		localOffset = dragon_li::util::prefixSumCta<THREADS, SizeType>(refineSize, 
				totalRefineSize);

		__shared__ SizeType globalOffset;

		if(threadIdx.x == 0 && totalRefineSize > 0) {
			globalOffset = ctaOutputAssignment.getCtaOutputAssignment(totalRefineSize);
		}

		__syncthreads();

		if(ctaOutputAssignment.getGlobalSize() > maxGridDataSize) //overflow
			return;


		DataType energy = 0;
		if(refineSize > 0) {
			devGridPointerStart[threadWorkOffset] = globalOffset + localOffset; //point to child cells
			energy = computeEnergy(gridData);
		}


		refineLevel++;
		if(refineLevel < maxRefineLevel) {

			SizeType cdpCtas = (refineSize + Settings::CDP_THREADS - 1) >> Settings::CDP_THREADS_BITS;
			amrCdpThreadRefineKernel
				<<< cdpCtas, CDP_THREADS >>> (
					refineSize,
					devGridData,
					devGridPointer,
					energy,
					maxGridDataSize,
					maxRefineLevel,
					globalOffset + localOffset,
					refineLevel + 1,
					ctaOutputAssignment
					);

		}

	}


	static __device__ void amrCdpRefineKernel(
	DataType * devGridData,
	SizeType * devGridPointer,
	SizeType maxGridDataSize,
	SizeType activeGridSize,
	SizeType maxRefineLevel,
	CtaOutputAssignment ctaOutputAssignment) {

		SizeType refineLevel = 0;

		CtaWorkAssignment ctaWorkAssignment(frontierSize);


		while(ctaWorkAssignment.workOffset < frontierSize) {
			ctaWorkAssignment.getCtaWorkAssignment();

			amrCdpCtaRefine(
					devGridData,
					devGridPointer,
					maxGridDataSize,
					maxRefineLevel,
					refineLevel);
		}


	}

};


template< typename Settings >
__global__ void amrCdpRefineKernel(
	typename Settings::DataType * devGridData,
	typename Settings::SizeType * devGridPointer,
	typename Settings::SizeType maxGridDataSize,
	typename Settings::SizeType activeGridSize,
	typename Settings::SizeType maxRefineLevel,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	AmrCdpDevice< Settings >::amrCdpRefineKernel(
					devGridData,
					devGridPointer,
					maxGridDataSize,
					activeGridSize,
					maxRefineLevel,
					ctaOutputAssignment);

}

}
}

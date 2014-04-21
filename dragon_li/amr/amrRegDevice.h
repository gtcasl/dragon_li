#pragma once

#include <dragon_li/util/primitive.h>
#include <dragon_li/util/ctaOutputAssignment.h>
#include <dragon_li/util/ctaWorkAssignment.h>

namespace dragon_li {
namespace amr {

template< typename Settings >
class AmrRegDevice {

	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;
	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;
	static const SizeType GRID_REFINE_SIZE = Settings::GRID_REFINE_SIZE;
	static const DataType GRID_REFINE_THRESHOLD = Settings::GRID_REFINE_THRESHOLD;
	static const DataType MAX_GRID_VALUE = Settings::MAX_GRID_VALUE;

	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;
	typedef typename dragon_li::util::CtaWorkAssignment<Settings> CtaWorkAssignment;

public:

public:
	static __device__ void amrRegCtaRefine(
		CtaWorkAssignment & ctaWorkAssignment,
		DataType * devGridData,
		SizeType * devGridPointer,
		SizeType processGridOffset,
		SizeType maxGridDataSize,
		SizeType maxRefineLevel,
		CtaOutputAssignment & ctaOutputAssignment) {

		DataType * devGridDataStart = devGridData + processGridOffset;
		SizeType * devGridPointerStart = devGridPointer + processGridOffset;
		DataType gridData;
		SizeType gridPointer = -1; //Not processed
		SizeType refineSize = 0;

		if(threadIdx.x < ctaWorkAssignment.workSize) {
			gridData = devGridDataStart[ctaWorkAssignment.workOffset + threadIdx.x];
			gridPointer = devGridPointerStart[ctaWorkAssignment.workOffset + threadIdx.x];

			if(gridPointer == -1) {
				if(gridData >= GRID_REFINE_THRESHOLD) {
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

		for(SizeType refineId = 0; refineId < refineSize; refineId++) {
			devGridData[globalOffset + localOffset + refineId] = gridData / refineSize; //random later 
		}


	}

	static __device__ void amrRegRefineKernel(
		DataType * devGridData,
		SizeType * devGridPointer,
		SizeType maxGridDataSize,
		SizeType processGridOffset,
		SizeType processGridSize,
		SizeType maxRefineLevel,
		CtaOutputAssignment & ctaOutputAssignment) {


		CtaWorkAssignment ctaWorkAssignment(processGridSize);
	
		while(ctaWorkAssignment.workOffset < processGridSize) {
			ctaWorkAssignment.getCtaWorkAssignment();
	
			amrRegCtaRefine(
				ctaWorkAssignment,
				devGridData,
				devGridPointer,
				processGridOffset,
				maxGridDataSize,
				maxRefineLevel,
				ctaOutputAssignment);
		}

	}
};

template< typename Settings >
__global__  void amrRegRefineKernel(
	typename Settings::DataType * devGridData,
	typename Settings::SizeType * devGridPointer,
	typename Settings::SizeType maxGridDataSize,
	typename Settings::SizeType processGridOffset,
	typename Settings::SizeType processGridSize,
	typename Settings::SizeType maxRefineLevel,
	typename dragon_li::util::CtaOutputAssignment<typename Settings::SizeType> ctaOutputAssignment) {

	AmrRegDevice< Settings >::amrRegRefineKernel(
		devGridData,
		devGridPointer,
		maxGridDataSize,
		processGridOffset,
		processGridSize,
		maxRefineLevel,
		ctaOutputAssignment);
}

}
}

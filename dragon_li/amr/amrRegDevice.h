#pragma once

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
		SizeType maxGridDataSize,
		SizeType maxRefineLevel,
		CtaOutputAssignment & ctaOutputAssignment) {
	}

	static __device__ void amrRegRefineKernel(
		DataType * devGridData,
		SizeType * devGridPointer,
		SizeType maxGridDataSize,
		SizeType activeGridSize,
		SizeType maxRefineLevel,
		CtaOutputAssignment & ctaOutputAssignment) {

		CtaWorkAssignment ctaWorkAssignment(activeGridSize);
	
		while(ctaWorkAssignment.workOffset < activeGridSize) {
			ctaWorkAssignment.getCtaWorkAssignment();
	
			amrRegCtaRefine(
				ctaWorkAssignment,
				devGridData,
				devGridPointer,
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
	typename Settings::SizeType activeGridSize,
	typename Settings::SizeType maxRefineLevel,
	typename dragon_li::util::CtaOutputAssignment<typename Settings::SizeType> ctaOutputAssignment) {

	AmrRegDevice< Settings >::amrRegRefineKernel(
		devGridData,
		devGridPointer,
		maxGridDataSize,
		activeGridSize,
		maxRefineLevel,
		ctaOutputAssignment);
}

}
}

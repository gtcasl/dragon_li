#pragma once

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace amr {

template< typename Settings >
__global__ void amrCdpThreadRefineKernel(
	typename Settings::SizeType refineSize,
	typename Settings::DataType * devGridData,
	typename Settings::SizeType * devGridPointer,
	typename Settings::DataType energy,
	typename Settings::SizeType maxGridDataSize,
	typename Settings::SizeType maxRefineLevel,
	typename Settings::SizeType outputOffset,
	typename Settings::SizeType refineLevel,
	typename dragon_li::util::CtaOutputAssignment< typename Settings::SizeType > ctaOutputAssignment) {

	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	SizeType refineId = threadIdx.x + blockIdx.x * blockDim.x;

	if(refineId < refineSize) {

		DataType refineData = computeTemperature(energy, refineId); 
		devGridData[globalOffset + localOffset + refineId] = refineData;
		reportDevice("%d.%d, offset %d, data %f\n", blockIdx.x, threadIdx.x, globalOffset + localOffset + refineId, refineData); 

	}


}

}
}

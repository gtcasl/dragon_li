#pragma once

#include <dragon_li/amr/amrBase.h>
#include <dragon_li/amr/amrRegDevice.h>

namespace dragon_li {
namespace amr {

template< typename Settings >
class AmrReg : public AmrBase< Settings > {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	AmrReg() : AmrBase<Settings>() {}	

	int refine() {

		amrRegRefineKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devGridData,
				this->devGridPointer,
				this->maxGridDataSize,
				this->activeGridSize,
				this->maxRefineLevel,
				this->ctaOutputAssignment
			);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		return 0;

	}
};

}
}

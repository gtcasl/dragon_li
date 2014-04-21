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

	SizeType iteration;

	AmrReg() : AmrBase<Settings>(),
		iteration(0) {}	

	int refine() {

		while(iteration < this->maxRefineLevel && !this->gridSizeOverFlow) {

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

			if(this->ctaOutputAssignment.getGlobalSize(this->activeGridSize))
				return -1;

			iteration++;

			if(iteration >= this->maxRefineLevel) {
				errorMsg("Max Refine Level reached! Consider increasing maxRefineLevel!");
			}
			if(this->activeGridSize > this->maxGridDataSize) {
				this->gridSizeOverflow = true;
				errorMsg("Grid Data Size overflow! Consider increasing maxGridDataSize!");
			}

		}

		return 0;

	}
};

}
}

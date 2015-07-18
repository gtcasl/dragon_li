#ifdef ENABLE_CDP

#pragma once

#include <dragon_li/amr/amrReg.h>
#include <dragon_li/amr/amrCdpDevice.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace amr {

template< typename Settings >
class AmrCdp : public AmrReg< Settings > {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	AmrCdp() : AmrReg< Settings >() {}

	int refine() {

		amrCdpRefineKernel< Settings >
				<<< CTAS, THREADS >>> (
					this->devGridData,
					this->devGridPointer,
					this->maxGridDataSize,
					this->activeGridSize,
					this->maxRefineLevel,
					this->gridRefineThreshold,
					this->ctaOutputAssignment
				);
		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		if(this->ctaOutputAssignment.getGlobalSize(this->activeGridSize))
			return -1;

        if(this->verbose)
            std::cout << "activeGridSize = " << this->activeGridSize << "\n";

		if(this->activeGridSize > this->maxGridDataSize) {
			this->gridSizeOverflow = true;
			errorMsg("Grid Data Size overflow! Consider increasing maxGridDataSize!");
			return -1;
		}


		return 0;


	}

};

}
}

#endif

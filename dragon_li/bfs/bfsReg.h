#pragma once

#include <dragon_li/bfs/bfsBase.h>
#include <dragon_li/bfs/bfsRegDevice.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsReg : public BfsBase< Settings > {

	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:

	BfsReg() : BfsBase< Settings >() {}

	int search() {
		while(this->frontierSize > 0) {
			report("Start BFS Search in regular mode... (" << CTAS << ", " << THREADS << ")");
			report("Iteration " << this->iteration );

			if(this->ctaOutputAssignment.reset())
				return -1;
	
			report("Expand...");
			expand();
		
			if(this->ctaOutputAssignment.getGlobalSize(this->frontierSize))
				return -1;
			
			if(this->ctaOutputAssignment.reset())
				return -1;

			report("Contract...");
			contract();

			if(this->ctaOutputAssignment.getGlobalSize(this->frontierSize))
				return -1;

			if(this->displayIteration(true))
				return -1;

//			//ping pong frontier buffer
//			SizeType * devTmp = this->devFrontierContract;
//			this->devFrontierContract = this->devFrontierExpand;
//			this->devFrontierExpand = devTmp;
	
			this->iteration++;

		}

		return 0;

	}

	int expand() {
				
		bfsRegExpandKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devColumnIndices,
				this->devRowOffsets,
				this->devFrontierContract,
				this->devFrontierExpand,
				this->maxFrontierSize,
				this->frontierSize,
				this->ctaOutputAssignment);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}

	int contract() {

		bfsRegContractKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devVisitedMasks,
				this->devFrontierExpand, //Frontier from expand
				this->devFrontierContract, //Output Frontier 
				this->maxFrontierSize,
				this->frontierSize,
				this->ctaOutputAssignment);

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

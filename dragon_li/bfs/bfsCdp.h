#pragma once

#include <dragon_li/bfs/bfsReg.h>
#include <dragon_li/bfs/bfsCdpDevice.h>

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsCdp : public BfsReg< Settings > {

	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:
	
	BfsCdp() : BfsReg< Settings >() {}

	int expand() {


		cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 8192);

		bfsCdpExpandKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devColumnIndices,
				this->devRowOffsets,
				this->devSearchDistance,
				this->devFrontierContract,
				this->devFrontierExpand,
				this->maxFrontierSize,
				this->frontierSize,
				this->ctaOutputAssignment,
				this->iteration);

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

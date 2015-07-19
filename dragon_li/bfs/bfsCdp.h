#ifdef ENABLE_CDP
#pragma once

#include <dragon_li/bfs/bfsReg.h>
#include <dragon_li/bfs/bfsCdpDevice.h>

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsCdp : public BfsReg< Settings > {

	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::SizeType SizeType;
	typedef typename dragon_li::util::GraphCsrDevice<typename Settings::Types> GraphCsrDevice;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:
	
	BfsCdp() : BfsReg< Settings >() {}

    int setup(
	    	GraphCsrDevice &graphCsrDevice,
			typename BfsReg<Settings>::UserConfig &userConfig) {
		return setup(
				graphCsrDevice.vertexCount,
				graphCsrDevice.edgeCount,
				graphCsrDevice.devColumnIndices,
				graphCsrDevice.devRowOffsets,
				userConfig
			);
	}

    int setup(
		SizeType _vertexCount,
		SizeType _edgeCount,
		VertexIdType * _devColumnIndices,
		SizeType * _devRowOffsets,
		typename BfsReg<Settings>::UserConfig & userConfig) {
       
        int status = 0;
 
        //Base class setup
        status = BfsReg< Settings >::setup(
            _vertexCount,
            _edgeCount,
            _devColumnIndices,
            _devRowOffsets,
            userConfig
        );
        if(status)
            return status;
		
        cudaError_t result = cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 131072);

        return 0;
    }

	int expand() {

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
#endif

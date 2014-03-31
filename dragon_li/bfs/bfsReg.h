#pragma once

#include <dragon_li/bfs/bfsBase.h>
#include <dragon_li/bfs/bfsRegDevice.h>

#undef REPORT_BASE
#define REPORT_BASE 1

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
		report("Start BFS Search in regular mode... (" << CTAS << ", " << THREADS << ")");
		bfsRegSearchKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devColumnIndices,
				this->devRowOffsets,
				this->devFrontierIn,
				this->devFrontierOut,
				this->maxFrontierSize,
				this->devFrontierSize,
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

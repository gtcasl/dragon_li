#pragma once

#include <algorithm>

#include <dragon_li/util/memsetDevice.h>
#include <dragon_li/join/joinBase.h>
#include <dragon_li/join/joinRegDevice.h>
#include <dragon_li/join/joinData.h>

#undef REPORT_BASE
#define REPORT_BASE 1 

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinReg : public JoinBase< Settings > {

public:
	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	//Processing temporary storage
	SizeType *devLowerBounds;
	SizeType *devUpperBounds;
	SizeType *devOutBounds;
	SizeType *devHistogram;
	SizeType *devJoinLeftOutIndicesScattered;
	SizeType *devJoinRightOutIndicesScattered;
	
	SizeType estJoinOutCount;


	JoinReg() : JoinBase< Settings >(), 
		devLowerBounds(NULL),
		devUpperBounds(NULL),
		devOutBounds(NULL),
		devHistogram(NULL),
		devJoinLeftOutIndicesScattered(NULL),
		devJoinRightOutIndicesScattered(NULL),
		estJoinOutCount(0) {}

	int setup(JoinData<Types> joinData,
				typename JoinBase<Settings>::UserConfig & userConfig) {

		//call setup from base class
		if(JoinBase<Settings>::setup(joinData, userConfig))
			return -1;

		cudaError_t retVal;
	
		if(retVal = cudaMalloc(&devLowerBounds, CTAS * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devUpperBounds, CTAS * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devOutBounds, (CTAS + 1) * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&devHistogram, (CTAS + 1) * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		estJoinOutCount = std::max(this->inputCountLeft, this->inputCountRight) * Settings::JOIN_SF; 

		if(retVal = cudaMalloc(&devJoinLeftOutIndicesScattered, estJoinOutCount * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMalloc(&devJoinRightOutIndicesScattered, estJoinOutCount * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}


		if(dragon_li::util::memsetDevice<Settings::CTAS, Settings::THREADS, DataType, SizeType>
			(devJoinLeftOutIndicesScattered, 0, estJoinOutCount))
			return -1;

		if(dragon_li::util::memsetDevice<Settings::CTAS, Settings::THREADS, DataType, SizeType>
			(devJoinRightOutIndicesScattered, 0, estJoinOutCount))
			return -1;


		return 0;
	
	}

	int findBounds() {
		
		joinRegFindBoundsKernel< Settings >
			<<< CTAS, THREADS>>> (
			this->devJoinInputLeft,
			this->inputCountLeft,
			this->devJoinInputRight,
			this->inputCountRight,
			devLowerBounds,
			devUpperBounds,
			devOutBounds
		);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}
		
		if(util::prefixScan<THREADS, DataType>(devOutBounds, CTAS + 1)) {
			errorMsg("Prefix Sum for outBounds fails");
			return -1;
		}

		return 0;
		
	}

	int mainJoin() {

		joinRegMainJoinKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devJoinInputLeft,
				this->inputCountLeft,
				this->devJoinInputRight,
				this->inputCountRight,
				devJoinLeftOutIndicesScattered,
				devJoinRightOutIndicesScattered,
				devHistogram,
				devLowerBounds,
				devUpperBounds,
				devOutBounds);

		cudaError_t retVal;
		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}

	int gather() {

		if(util::prefixScan<THREADS, DataType>(devHistogram, CTAS + 1)) {
			errorMsg("Prefix Sum for histogram fails");
			return -1;
		}

		cudaError_t retVal;
		if(retVal = cudaMemcpy(&this->outputCount, devHistogram + CTAS, sizeof(SizeType), cudaMemcpyDeviceToHost)) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&this->devJoinLeftOutIndices, this->outputCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		if(retVal = cudaMalloc(&this->devJoinRightOutIndices, this->outputCount * sizeof(SizeType))) {
			errorCuda(retVal);
			return -1;
		}

		joinRegGatherKernel< Settings >
			<<< CTAS, THREADS >>> (
				this->devJoinLeftOutIndices,
				this->devJoinRightOutIndices,
				devJoinLeftOutIndicesScattered,
				devJoinRightOutIndicesScattered,
				estJoinOutCount,
				devOutBounds,
				devHistogram,
				this->devJoinOutputCount
			);

		if(retVal = cudaDeviceSynchronize()) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}

	int join() {

		report("Finding bounds...");
		if(findBounds())
			return -1;

		report("Main Join...");

		if(mainJoin())
			return -1;

		report("Gathering...");
		
		if(gather())
			return -1;

		return 0;
			
	}
};

}
}

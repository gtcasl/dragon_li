#pragma once

#include <algorithm>
#include <vector>

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

	int setup(JoinData<Types> & joinData,
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
			<<< (CTAS + THREADS - 1)/THREADS, THREADS >>> (
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

		if(this->veryVerbose) {
			std::vector<SizeType> upper(CTAS), lower(CTAS);
			cudaMemcpy(lower.data(), devLowerBounds, CTAS * sizeof(SizeType), cudaMemcpyDeviceToHost);
			cudaMemcpy(upper.data(), devUpperBounds, CTAS * sizeof(SizeType), cudaMemcpyDeviceToHost);
		

			std::cout << "Right Bounds:\n";
			for(int i = 0; i < CTAS; i++)
				std::cout << "[" << lower[i] << ", " << upper[i] << "], ";
			std::cout << "\n\n";

		}
		
		if(util::prefixScan<THREADS, SizeType>(devOutBounds, CTAS + 1)) {
			errorMsg("Prefix Sum for outBounds fails");
			return -1;
		}

		if(this->veryVerbose) {
			std::vector<SizeType> outBounds(CTAS+1);
			cudaMemcpy(outBounds.data(), devOutBounds, (CTAS+1) * sizeof(SizeType), cudaMemcpyDeviceToHost);
			std::cout << "Out Bounds:\n";
			for(int i = 0; i < CTAS+1; i++)
				std::cout << outBounds[i] << ", ";
			std::cout << "\n\n";
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

//		std::vector<SizeType> upper(1200);
//		cudaMemcpy(upper.data(), devJoinRightOutIndicesScattered, (1200) * sizeof(SizeType), cudaMemcpyDeviceToHost);
//		for(int i = 0; i < 1200; i++)
//			std::cout << "u" << i << ": " << upper[i] << "\n";
		

		return 0;
	}

	int gather() {

		if(util::prefixScan<THREADS, DataType>(devHistogram, CTAS + 1)) {
			errorMsg("Prefix Sum for histogram fails");
			return -1;
		}
		if(this->veryVerbose) {
			std::vector<SizeType> histogram(CTAS+1);
			cudaMemcpy(histogram.data(), devHistogram, (CTAS+1) * sizeof(SizeType), cudaMemcpyDeviceToHost);
			std::cout << "Histogram:\n";
			for(int i = 0; i < CTAS+1; i++)
				std::cout << histogram[i] << ", ";
			std::cout << "\n\n";
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

		std::cout << "Join output size " << this->outputCount << "\n";

		if(this->veryVerbose) {
			std::vector<SizeType> leftIndices(this->outputCount), rightIndices(this->outputCount);
			cudaMemcpy(leftIndices.data(), this->devJoinLeftOutIndices, 
				this->outputCount * sizeof(DataType), 
				cudaMemcpyDeviceToHost);
			cudaMemcpy(rightIndices.data(), this->devJoinRightOutIndices, 
				this->outputCount * sizeof(DataType), 
				cudaMemcpyDeviceToHost);
		

			std::cout << "Output Indices:\n";
			for(int i = 0; i < this->outputCount; i++)
				std::cout << "[" << leftIndices[i] << ", " << rightIndices[i] << "], ";
			std::cout << "\n\n";

		}

		return 0;
	}

	int join() {

		if(this->verbose)
			std::cout << "Finding bounds...\n";

		if(findBounds())
			return -1;

		if(this->verbose)
			std::cout << "Main Join...\n";

		if(mainJoin())
			return -1;

		if(this->verbose)
			std::cout << "Gathering...\n";
		
		if(gather())
			return -1;

		return 0;
			
	}
};

}
}

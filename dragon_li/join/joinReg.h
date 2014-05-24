#pragma once

#include <algorithm>

#include <dragon_li/util/memsetDevice.h>
#include <dragon_li/join/joinBase.h>
#include <dragon_li/join/joinRegDevice.h>

#undef REPORT_BASE
#define REPORT_BASE 1 

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinReg : public JoinBase< Settings > {

	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

public:

	//Processing temporary storage
	SizeType *devLowerBounds;
	SizeType *devUpperBounds;
	SizeType *devOutBounds;
	SizeType *devHistogram;
	SizeType *devJoinOutputScattered;
	
	SizeType estJoinOutCount;


	JoinReg() : JoinBase< Settings >(), 
		devLowerBounds(NULL),
		devUpperBounds(NULL),
		devOutBounds(NULL),
		devHistogram(NULL),
		devJoinOutputScattered(NULL),
		estJoinOutCount(0) {}

	int findBound() {
	
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

		estJoinOutCount = std::max(this->inputCountLeft, this->inputCountRight) * this->joinEstOutScaleFactor; 

		if(retVal = cudaMalloc(&devJoinOutputScattered, estJoinOutCount * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}


		if(dragon_li::util::memsetDevice<Settings::CTAS, Settings::THREADS, DataType, SizeType>
			(devJoinOutputScattered, 0, estJoinOutCount))
			return -1;
			

		
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

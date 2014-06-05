#pragma once


#include <dragon_li/util/userConfig.h>
#include <dragon_li/util/debug.h>
#include <dragon_li/join/joinData.h>

#undef REPORT_BASE
#define REPORT_BASE

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinBase {

public:

	typedef typename Settings::Types Types;
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	class UserConfig : public dragon_li::util::UserConfig {
	public:

		UserConfig(
			bool _verbose,
			bool _veryVerbose
			) :
				dragon_li::util::UserConfig(_verbose, _veryVerbose)
				{}
	};



	//User control
	bool verbose;
	bool veryVerbose;

	//Join information
	SizeType inputCountLeft;
	SizeType inputCountRight;
	SizeType outputCount;

	//Join Device information
	DataType * devJoinInputLeft;
	DataType * devJoinInputRight;
	SizeType * devJoinLeftOutIndices;
	DataType * devJoinRightOutIndices;
	SizeType * devJoinOutputCount;

	JoinBase() : 
		verbose(false),
		veryVerbose(false),
		inputCountLeft(0),
		inputCountRight(0),
		outputCount(0),
		devJoinInputLeft(NULL),
		devJoinInputRight(NULL),
		devJoinLeftOutIndices(NULL),
		devJoinRightOutIndices(NULL),
		devJoinOutputCount(NULL) {}

	virtual int join() = 0;

	virtual int setup(JoinData<Types> joinData,
					UserConfig & userConfig) {

		verbose = userConfig.verbose;
		veryVerbose = userConfig.veryVerbose;

		inputCountLeft = joinData.inputCountLeft;
		inputCountRight = joinData.inputCountRight;

		cudaError_t retVal;

		if(retVal = cudaMalloc(&devJoinInputLeft, inputCountLeft * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(devJoinInputLeft, 
								joinData.inputLeft.data(), 
								inputCountLeft * sizeof(DataType),
								cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMalloc(&devJoinInputRight, inputCountRight * sizeof(DataType))) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(devJoinInputRight, 
								joinData.inputRight.data(), 
								inputCountRight * sizeof(DataType),
								cudaMemcpyHostToDevice)) {
			errorCuda(retVal);
			return -1;
		}

		return 0;
	}

	virtual int displayResult() {
		return 0;
	}

	virtual int finish() {
		return 0;
	}
};

}
}

#pragma once

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
	DataType * devJoinOutput;
	SizeType * devJoinOutputCount;

	JoinBase() : 
		verbose(false),
		veryVerbose(false),
		inputCountLeft(0),
		inputCountRight(0),
		outputCount(0),
		devjoinInputLeft(NULL),
		devJoinInputRight(NULL),
		devJoinOutput(NULL),
		devJoinOutputCount(NULL) {}

	virtual int join() = 0;

	virtual int setup(JoinData<Types> joinData) {

		cudaMalloc(&devJoininputLeft
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

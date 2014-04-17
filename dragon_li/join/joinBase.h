#pragma once

#undef REPORT_BASE
#define REPORT_BASE

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinBase {

public:

	typedef Settings::SizeType SizeType;


	//User control
	bool verbose;
	bool veryVerbose;

	SizeType inputCount;

	virtual int join() = 0;

	virtual int setup() {
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

#pragma once

#undef REPORT_BASE
#define REPORT_BASE

namespace dragon_li {
namespace join {

template< typename Types >
class JoinData {

public:
	typedef typename Types::SizeType SizeType;
	typedef typename Types::DataType DataType;

	SizeType inputCountLeft;
	SizeType inputCountRight;

	std::vector<DataType> inputLeft;
	std::vector<DataType> inputRight;

	JoinData() : 
		inputCountLeft(0),
		inputCountRight(0) {}

	int generateRandomData(SizeType countLeft, SizeType countRight) {
		
		inputCountLeft = countLeft;
		inputCountRight = countRight;

		inputLeft.resize(countLeft);
		inputRight.resize(countRight);

		dragon_li::util::random<DataType>(inputLeft, countLeft, 0, countLeft);
		dragon_li::util::random<DataType>(inputRight, countRight, 0, countRight);

		return 0;

	}

	int readFromDataFile() {
		return 0;
	}

};

}
}

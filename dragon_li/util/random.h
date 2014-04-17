#pragma once 

#include <cstdlib>
#include <ctime>

namespace dragon_li {
namespace util {

template<typename DataType, typename SizeType>
int random(std::vector<DataType> & data, 
			SizeType count,
			DataType rangeStart,
			DataType rangeEnd) {

	errorMsg("Not implemented");
}

template<typename int, typename SizeType>
int random(std::vector<int> & data, 
			SizeType count,
			int rangeStart,
			int rangeEnd) {

	std::srand(std::time(0));
	for(SizeType i = 0; i < count; i++) {
		int randomNum = std::rand() % (rangeEnd - rangeStart) + rangeStart;
		data[i] = randomNum;

	}

}

#pragma once

#include <list>

namespace dragon_li {
namespace amr {

template< typename Types >
class AmrCpu {
	typedef typename Types::SizeType SizeType;
	typedef typename Types::DataType DataType;

public:
	class CpuConfig : public dragon_li::util::UserConfig {
		DataType startGridValue;
		DataType gridRefineThreshold;

		CpuConfig(
			bool _verbose,
			bool _veryVerbose,
			DataType _startGridValue,
			DataType _gridRefineThreshold) :
				util::UserConfig(_verbose, _veryVerbose),
				startGridValue(_startGridValue),
				gridRefineThreshold(_gridRefineThreshold) {}
	};

	struct AmrCpuData {
		DataType data;
		typename std::list<AmrCpuData>::iterator childPtr;

		AmrCpuData(DataType _data,
			typename std::list<AmrCpuData>::iterator _childPtr) {

			data = _data;
			childPtr = _childPtr;
		}
	};

	static std::list<AmrCpuData> cpuAmrData;

	static int amrCpu(DataType startGridValue) {

		cpuAmrData.push_back(AmrCpuData(startGridValue, cpuAmrData.end()));

		typename std::list<AmrCpuData>::iterator curIt = cpuAmrData.begin();
		while (curIt != cpuAmrData.end()) {
			DataType gridData = curIt->data;
		//	if(gridData > 
		}

		return 0;
		

	}
};

template<typename Types>
std::list<typename AmrCpu<Types>::AmrCpuData> AmrCpu<Types>::cpuAmrData;
}
}

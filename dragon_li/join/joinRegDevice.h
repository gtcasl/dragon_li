#pragma once

#include <dragon_li/join/joinReg.h>
#include <dragon_li/util/threadWorkAssignment.h>

#undef REPORT_BASE
#define REPORT_BASE 1

namespace dragon_li {
namespace join {

template< typename Settings >
class JoinRegDevice {
	typedef typename Settings::SizeType SizeType;
	typedef typename Settings::DataType DataType;

	static const SizeType THREADS = Settings::THREADS;
	static const SizeType CTAS = Settings::CTAS;

	typedef typename dragon_li::util::ThreadWorkAssignment<Settings> ThreadWorkAssignment;

public:

	static __device__ void joinRegThreadFindBounds(
		threadWorkAssignment & threadWorkAssignment,
		DataType * devJoinInputLeft,
		SizeType inputCountLeft,
		DataType * devJoinInputRight,
		SizeType inputCountRight,
		SizeType * devLowerBounds,
		SizeType * devUpperBounds,
		SizeType * devOutBounds
	) {

		if(workSize == 0)
			return;

			
		
	}

	static __device__ void joinRegFindBoundsKernel(
		DataType * devJoinInputLeft,
		SizeType inputCountLeft,
		DataType * devJoinInputRight,
		SizeType inputCountRight,
		SizeType * devLowerBounds,
		SizeType * devUpperBounds,
		SizeType * devOutBounds
	) {
	
		SizeType partitions = CTAS;
		SizeType partitionSize = (inputCountLeft + partitions - 1) / partitions;

		ThreadWorkAssignment threadWorkAssignment(partitions);
		
		while(threadWorkAssignment.workOffset < partitions) {
			threadWorkAssignment.getThreadWorkAssignment();

			joinRegThreadFindBound(
				threadWorkAssignment,
				devJoinInputLeft,
				inputCountLeft,
				devJoinInputRight,
				inputCountRight,
				devLowerBounds,
				devUpperBounds,
				devOutBounds
			);
		}
	}

};

template< typename Settings >
__global__ void joinRegFindBoundsKernel(
	typename Settings::DataType * devJoinInputLeft,
	typename Settings::SizeType inputCountLeft,
	typename Settings::DataType * devJoinInputRight,
	typename Settings::SizeType inputCountRight,
	typename Settings::SizeType * devLowerBounds,
	typename Settings::SizeType * devUpperBounds,
	typename Settings::SizeType * devOutBounds
	) {

	JoinRegDevice< Settings >::joinRegFindBoundsKernel (
			devJoinInputLeft,
			inputCountLeft,
			devJoinInputRight,
			inputCountRight,
			devLowerBounds,
			devUpperBounds,
			devOutBounds
	);


}

}
}

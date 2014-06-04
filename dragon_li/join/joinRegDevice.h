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

	static SizeType joinBlockEstOutScaleFactor;  

	static __device__ int joinRegThreadFindLowerBound(
		DataType key,
		DataType * data,
		DataType * dataCount
	) {
		SizeType low = 0;
		SizeType high = dataCount;

		SizeType mid = low + (high - low) / 2;
		if(begin[mid] < key)
			low = mid + 1;
		else
			high = mid;

		return low;
	}

	static __device__ void joinRegThreadFindBounds(
		threadWorkAssignment & threadWorkAssignment,
		SizeType partitionSize,
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

		SizeType partitionId = threadWorkAssignment.workOffset;
		SizeType leftStart = min(partitionSize * partitionId, inputCountLeft);
		SizeType leftEnd = min(partitionSize * (partitionId + 1), inputCountLeft);

		if(leftStart < leftEnd) {
			
			DataType lowerKey = devJoinInputLeft[leftStart];
			SizeType lowerBound = joinRegThreadFindLowerBound(
						lowerKey,
						devJoinInputRight,
						inputCountRight);

			devLowerBounds[partitionId] = lowerBound;

			DataType upperKey = devJoinInputLeft[leftEnd - 1];
			SizeType upperBound = joinREgThreadFindUpperBound(
						upperkey,
						devJoinInputRight,
						inputCountRight);
			devUpperBounds[partitionId] = upperBound;
			devOutBounds[partitionId] = max(upperBounds - lowerBound, leftEnd - leftStart) * joinBlockEstOutScaleFactor;
		}
		else {
			//out of bound
			devLowerBounds[partitionId] = inputCountRight;
			devUpperBounds[partitionId] = inputCountRight;
			devOutBounds[partitionId] = 0;
		}
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
				partitionSize,
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

	static __device__ void joinRegMainJoinKernel(
		DataType * devJoinInputLeft,
		SizeType inputCountLeft,
		DataType * devJoinInputRight,
		SizeType inputCountRight,
		SizeType * devJoinLeftOutIndicesScattered,
		SizeType * devJoinRightOutIndicesScattered,
		SizeType * devHistogram,
		SizeType * devLowerBounds,
		SizeType * devUpperBounds,
		SizeType * devOutBounds
	) {
	}

	static __device__ void joinRegGatherKernel(
		SizeType * devJoinLeftOutIndices,
		SizeType * devJoinRightOutIndices,
		SizeType * devJoinLeftOutIndicesScattered,
		SizeType * devJoinRightOutIndicesScattered,
		SizeType * estJoinOutCount,
		SizeType * devOutBounds,
		SizeType * devHistogram,
		SizeType devJoinOutputCount
	) {
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

template< typename Settings >
__global__ void joinRegMainJoinKernel(
	typename Settings::DataType * devJoinInputLeft,
	typename Settings::SizeType inputCountLeft,
	typename Settings::DataType * devJoinInputRight,
	typename Settings::SizeType inputCountRight,
	typename Settings::SizeType * devJoinLeftOutIndicesScattered,
	typename Settings::SizeType * devJoinRightOutIndicesScattered,
	typename Settings::SizeType * devHistogram,
	typename Settings::SizeType * devLowerBounds,
	typename Settings::SizeType * devUpperBounds,
	typename Settings::SizeType * devOutBounds
	) {

	JoinRegDevice< Settings >::joinRegMainJoinKernel(
		devJoinInputLeft,
		inputCountLeft,
		devJoinInputRight,
		inputCountRight,
		devJoinLeftOutIndicesScattered,
		devJoinRightOutIndicesScattered,
		devHistogram,
		devLowerBounds,
		devUpperBounds,
		devOutBounds);
}

template< typename Settings >
__global__ void joinRegGatherKernel(
	typename Settings::SizeType * devJoinLeftOutIndices,
	typename Settings::SizeType * devJoinRightOutIndices,
	typename Settings::SizeType * devJoinLeftOutIndicesScattered,
	typename Settings::SizeType * devJoinRightOutIndicesScattered,
	typename Settings::SizeType * estJoinOutCount,
	typename Settings::SizeType * devOutBounds,
	typename Settings::SizeType * devHistogram,
	typename Settings::SizeType devJoinOutputCount
	) {
	JoinRegDevice< Settings >::joinRegGatherKernel (
		devJoinLeftOutIndices,
		devJoinRightOutIndices,
		devJoinLeftOutIndicesScattered,
		devJoinRightOutIndicesScattered,
		estJoinOutCount,
		devOutBounds,
		devHistogram,
		devJoinOutputCount
	);


}

}
}

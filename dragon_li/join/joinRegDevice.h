#pragma once

#include <dragon_li/join/joinReg.h>
#include <dragon_li/util/threadWorkAssignment.h>
#include <dragon_li/util/primitive.h>

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
	static const SizeType JOIN_BLOCK_SF = Settings::JOIN_BLOCK_SF;  
    static const SizeType JOIN_BLOCK_CACHE_SIZE = THREADS;

	typedef typename dragon_li::util::ThreadWorkAssignment<Settings> ThreadWorkAssignment;

public:


	static __device__ int joinRegThreadFindLowerBound(
		const DataType key,
		const DataType * data,
		const SizeType dataCount
	) {
		SizeType low = 0;
		SizeType high = dataCount;

		while(low < high) {
			SizeType mid = low + (high - low) / 2;
			if(data[mid] < key)
				low = mid + 1;
			else
				high = mid;
		}

		return low;
	}

	static __device__ int joinRegThreadFindUpperBound(
		const DataType key,
		const DataType * data,
		const SizeType dataCount
	) {
		SizeType low = 0;
		SizeType high = dataCount;

		SizeType mid = low + (high - low) / 2;
		if(key < data[mid])
			high = mid;
		else
			low = mid + 1;

		return low;
	}


	static __device__ void joinRegThreadFindBounds(
		ThreadWorkAssignment & threadWorkAssignment,
		const SizeType partitionSize,
		const DataType * devJoinInputLeft,
		const SizeType inputCountLeft,
		const DataType * devJoinInputRight,
		const SizeType inputCountRight,
		SizeType * devLowerBounds,
		SizeType * devUpperBounds,
		SizeType * devOutBounds
	) {

		if(threadWorkAssignment.workSize == 0)
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
			SizeType upperBound = joinRegThreadFindUpperBound(
						upperKey,
						devJoinInputRight,
						inputCountRight);
			devUpperBounds[partitionId] = upperBound;
			devOutBounds[partitionId] = max(upperBound - lowerBound, leftEnd - leftStart) * JOIN_BLOCK_SF;
		}
		else {
			//out of bound
			devLowerBounds[partitionId] = inputCountRight;
			devUpperBounds[partitionId] = inputCountRight;
			devOutBounds[partitionId] = 0;
		}
	}

	static __device__ void joinRegFindBoundsKernel(
		const DataType * devJoinInputLeft,
		const SizeType inputCountLeft,
		const DataType * devJoinInputRight,
		const SizeType inputCountRight,
		SizeType * devLowerBounds,
		SizeType * devUpperBounds,
		SizeType * devOutBounds
	) {
	
		SizeType partitions = CTAS;
		SizeType partitionSize = (inputCountLeft + partitions - 1) / partitions;

		ThreadWorkAssignment threadWorkAssignment(partitions);
		
		while(threadWorkAssignment.workOffset < partitions) {
			threadWorkAssignment.getThreadWorkAssignment();

			joinRegThreadFindBounds(
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

    static __device__ SizeType joinRegJoinBlock(
        SizeType* outLeft, 
        SizeType * outRight,
    	const SizeType leftStartId, 
        const DataType* left,  
        const SizeType leftElements,
    	const SizeType rightStartId, 
        const DataType* right, 
        const SizeType rightElements)
    {
    	__shared__ SizeType cacheLeft[JOIN_BLOCK_CACHE_SIZE];
    	__shared__ SizeType cacheRight[JOIN_BLOCK_CACHE_SIZE];
    
    
    	const DataType* r = right + threadIdx.x;
    	
    	DataType rKey = 0;
    	SizeType foundCount = 0;	
    	
        SizeType lower = 0;
    	SizeType higher = 0;
    
    	if(threadIdx.x < rightElements)
    	{
    		rKey = *r;
    
    		lower  = joinRegThreadFindLowerBound(rKey, left, leftElements);
    		higher = joinRegThreadFindUpperBound(rKey, left, leftElements);
    		
    		foundCount = higher - lower;
    	}
    	
    	SizeType total = 0;
    	SizeType index = util::prefixSumCta<THREADS, DataType>(foundCount, total);
    
    	__syncthreads();
    	
    	if(total <= JOIN_BLOCK_CACHE_SIZE)
    	{
    		for(SizeType c = 0; c < foundCount; ++c)
    		{
//    			DataType lValue = left[lower+c];
                SizeType leftId = leftStartId + lower + c;
    			cacheLeft[index + c] = leftId;

                SizeType rightId = rightStartId + threadIdx.x;
                cacheRight[index + c] = rightId;
    		}
    		
    		__syncthreads();
    		util::memcpyCta<THREADS, DataType>(outLeft, cacheLeft, total);
    		util::memcpyCta<THREADS, DataType>(outRight, cacheRight, total);
    
    	}
    	else
    	{
    		__shared__ SizeType sharedCopiedThisTime;
    	
    		SizeType copiedSoFar = 0;
    		bool done = false;
    	
    		while(copiedSoFar < total)
    		{
    			if(index + foundCount <= JOIN_BLOCK_CACHE_SIZE && !done)
    			{
    				for(SizeType c = 0; c < foundCount; ++c)
    				{
//    					DataType lValue = left[lower + c];
//    					cache[index + c] = lValue; 
                        SizeType leftId = leftStartId + lower + c;
            			cacheLeft[index + c] = leftId;

                        SizeType rightId = rightStartId + threadIdx.x;
                        cacheRight[index + c] = rightId;
    				}
    			
    				done = true;
    			}
    			
    			if(index <= JOIN_BLOCK_CACHE_SIZE && index + foundCount > JOIN_BLOCK_CACHE_SIZE) //overbounded thread
    				sharedCopiedThisTime = index;
       			else if (threadIdx.x == THREADS - 1 && done)
    	           	sharedCopiedThisTime = index + foundCount;
    			
    			__syncthreads();
    		
    			SizeType copiedThisTime = sharedCopiedThisTime;
    		
    			index -= copiedThisTime;
    			copiedSoFar += copiedThisTime;
    		
    			outLeft = util::memcpyCta<THREADS, DataType>(outLeft, cacheLeft, copiedThisTime);
    			outRight = util::memcpyCta<THREADS, DataType>(outRight, cacheRight, copiedThisTime);
    	        
    		}
    
    	}
    
    	return total;
    }

	static __device__ void joinRegMainJoinKernel(
		const DataType * devJoinInputLeft,
		const SizeType inputCountLeft,
		const DataType * devJoinInputRight,
		const SizeType inputCountRight,
		SizeType * devJoinLeftOutIndicesScattered,
		SizeType * devJoinRightOutIndicesScattered,
		SizeType * devHistogram,
		const SizeType * devLowerBounds,
		const SizeType * devUpperBounds,
		const SizeType * devOutBounds
	) {
    	__shared__ DataType leftCache[THREADS];
    	__shared__ DataType rightCache[THREADS];
    
    	SizeType id = blockIdx.x;
    	
		SizeType partitions = CTAS;
		SizeType partitionSize = (inputCountLeft + partitions - 1) / partitions;
    	
    	SizeType leftId = MIN(partitionSize * id, inputCountLeft);
        const DataType* l    = devJoinInputLeft + leftId;
    	const DataType* lend = devJoinInputLeft + MIN(partitionSize * (id + 1), inputCountLeft);
    
   
        SizeType rightId = devLowerBounds[id];
    	const DataType* r    = devJoinInputRight + rightId;
    	const DataType* rend = devJoinInputRight + devUpperBounds[id];
    
    	
    	SizeType* oBeginLeft = devJoinLeftOutIndicesScattered + devOutBounds[id] - devOutBounds[0];
    	SizeType* oLeft      = oBeginLeft;
    	SizeType* oBeginRight = devJoinRightOutIndicesScattered + devOutBounds[id] - devOutBounds[0];
    	SizeType* oRight      = oBeginRight;
    
    	//SizeType joined_total = 0;
    
    	while(l != lend && r != rend)
    	{
    		SizeType leftBlockSize  = MIN(lend - l, THREADS);
    		SizeType rightBlockSize = MIN(rend - r, THREADS);
    
    		util::memcpyCta<THREADS, DataType>(leftCache,  l, leftBlockSize);
    		util::memcpyCta<THREADS, DataType>(rightCache, r, rightBlockSize);
    
    		__syncthreads();
    
    		DataType lMaxValue = *(leftCache + leftBlockSize - 1);
    		DataType rMinValue = *rightCache;
    	
    		if(lMaxValue < rMinValue)
    		{
    //				if(debug && threadIdx.x == 0) {
    //					printf("lMax = %u,rMin = %u\n", lMaxValue,(rMinValue) );
    //					printf("%u, joined total for block = %d\n", blockIdx.x, joined_total);
    //					printf("%u\n", sizeof(DataType));
    //
    //					for(SizeType iii = 0; iii < leftBlockSize; iii++)
    //						printf("awu %u\n", extract<SizeType, 0, Left>(l[iii]));
    //			
    //					for(SizeType iii = 0; iii < leftBlockSize; iii++)
    //						printf("cache %u\n", extract<SizeType, 0, Left>(leftCache[iii]));
    //				}
    //
    
                leftId += leftBlockSize;
    			l += leftBlockSize;
    //			joined_total = 0;
    		}
    		else
    		{
    			DataType lMinValue = *leftCache;
    			DataType rMaxValue = *(rightCache + rightBlockSize - 1);
    			
    			if(rMaxValue < lMinValue)
    			{
                    rightId += rightBlockSize;
    				r += rightBlockSize;
    			}
    			else
    			{
    		//		if(blockIdx.x == 8 && threadIdx.x == 0 && l >= lend-144-256) {
    		//			for(int iii = 0; iii < leftBlockSize; iii++)
    		//				printf("left %d = %llu\n", iii, stripValues<Left, keyFields>(l[iii]));
    		//			for(int iii = 0; iii < leftBlockSize; iii++)
    		//				printf("right %d = %llu\n", iii, stripValues<Right, keyFields>(r[iii]));
    		//		}
    
    
    
    				SizeType joined = joinRegJoinBlock(oLeft, oRight,
    					leftId, leftCache,  leftBlockSize,
    					rightId, rightCache, rightBlockSize);
    			//	if(blockIdx.x == 8 && threadIdx.x == 0 && l >= lend-144-256)
    			//		printf("joined %d\n", joined);
    
                   //if(joined > 0) //FIX ME
    				{	
    //					joined_total += joined;			
    					oLeft += joined;
                        oRight += joined;

                        SizeType rId = rightId + rightBlockSize;
    					const DataType* ri = r + rightBlockSize;
    			
    					for(; ri != rend;)
    					{
    						rightBlockSize = MIN(THREADS, rend - ri);
    						util::memcpyCta<THREADS, DataType>(rightCache, ri, rightBlockSize);
    					
    						__syncthreads();
    						rMinValue = *rightCache;
    //						rMinValue = extract<SizeType,0, Right>(*rightCache);
    	
       				
    						if(lMaxValue < rMinValue) break;
    			
    //					if(blockIdx.x == 8 && threadIdx.x == 0 && l >= lend-144-256) {
    //					for(int iii = 0; iii < leftBlockSize; iii++)
    //						printf("left %d = %llu\n", iii, stripValues<Left, keyFields>(l[iii]));
    //					for(int iii = 0; iii < leftBlockSize; iii++)
    //						printf("right %d = %llu\n", iii, stripValues<Right, keyFields>(r[iii]));
    //				}
    
    						joined = joinRegJoinBlock(oLeft, oRight,
    							leftId, leftCache,  leftBlockSize,
    							rId, rightCache, rightBlockSize);
    	                           
    
    				//	if(blockIdx.x == 8 && threadIdx.x == 0 && l >= lend-144-256)
    				//		printf("joined %d\n", joined);
    						
    //						joined_total += joined;
    						oLeft += joined;
                            oRight += joined;

                            rId += rightBlockSize;
    						ri += rightBlockSize;
    
    					}
    				}
            	
                    leftId += leftBlockSize;
    				l += leftBlockSize;
    //				if(debug && blockIdx.x == 70 && threadIdx.x == 0)
    //					printf("lMax = %u, rMin = %u\n", (lMaxValue), (rMinValue));
    //				printf("joined total for block = %d\n", joined_total);
    //				joined_total = 0;
    			}
    			__syncthreads();
    		}
    	}
    
    
    	if(threadIdx.x == 0) {devHistogram[id] = oLeft - oBeginLeft;/* if(debug && id == 70) printf("%u,histogram=%u\n",id,histogram[id]);*/}

	}

	static __device__ void joinRegGatherKernel(
		SizeType * devJoinLeftOutIndices,
		SizeType * devJoinRightOutIndices,
		SizeType * devJoinLeftOutIndicesScattered,
		SizeType * devJoinRightOutIndicesScattered,
		SizeType estJoinOutCount,
		SizeType * devOutBounds,
		SizeType * devHistogram,
		SizeType * devJoinOutputCount
	) {
	}

};

template< typename Settings >
__global__ void joinRegFindBoundsKernel(
	const typename Settings::DataType * devJoinInputLeft,
	const typename Settings::SizeType inputCountLeft,
	const typename Settings::DataType * devJoinInputRight,
	const typename Settings::SizeType inputCountRight,
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
	const typename Settings::DataType * devJoinInputLeft,
	const typename Settings::SizeType inputCountLeft,
	const typename Settings::DataType * devJoinInputRight,
	const typename Settings::SizeType inputCountRight,
	typename Settings::SizeType * devJoinLeftOutIndicesScattered,
	typename Settings::SizeType * devJoinRightOutIndicesScattered,
	typename Settings::SizeType * devHistogram,
	const typename Settings::SizeType * devLowerBounds,
	const typename Settings::SizeType * devUpperBounds,
	const typename Settings::SizeType * devOutBounds
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
	typename Settings::SizeType estJoinOutCount,
	typename Settings::SizeType * devOutBounds,
	typename Settings::SizeType * devHistogram,
	typename Settings::SizeType * devJoinOutputCount
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

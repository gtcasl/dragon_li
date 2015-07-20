#pragma once

namespace dragon_li {
namespace util {

// A very straight-forward PrefixSum with bad performance
template< int CTA_SIZE, typename DataType >
__device__ DataType prefixSumCta(DataType input, DataType &total, DataType carryIn = 0) {

	__shared__ DataType sharedMem[CTA_SIZE];

	if(threadIdx.x < CTA_SIZE)
		sharedMem[threadIdx.x] = input;

	__syncthreads();

	for(int step = 1; step < CTA_SIZE; step <<= 1) {
		if(threadIdx.x >= step)
			input = input + sharedMem[threadIdx.x - step];
		__syncthreads();
		sharedMem[threadIdx.x] = input;
		__syncthreads();
	}

	total = sharedMem[CTA_SIZE - 1] + carryIn;

	if(threadIdx.x == 0)
		return carryIn;
	else
		return sharedMem[threadIdx.x - 1] + carryIn;
}

template< int CTA_SIZE, typename DataType >
__device__ DataType * memcpyCta(DataType *output, const DataType *input, int count) {

    for(int i = threadIdx.x; i < count; i+= CTA_SIZE)
        output[i] = input[i];

    return output + count;
}

}
}

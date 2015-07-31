#pragma once

#include <stdio.h>
#include <hydrazine/interface/debug.h>

#define errorMsg(x) \
	std::cout << "(" << hydrazine::_debugTime() << ") " \
		<< hydrazine::_debugFile( __FILE__, __LINE__ ) \
		<< " Error: " << x << "\n";

#define errorCuda(x) errorMsg(cudaGetErrorString(x))

#ifndef NDEBUG
	#define reportDevice(...) \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL) { \
			printf(__VA_ARGS__); \
		}
#else
	#define reportDevice(...)
#endif


#ifndef NDEBUG
	#define checkErrorDevice() { \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL) { \
			cudaError_t retVal = cudaGetLastError(); \
			if(retVal) \
				printf(cudaGetErrorString(retVal)); \
		}\
	}
#else
	#define checkErrorDevice()
#endif

namespace dragon_li {
namespace util {

#ifndef NDEBUG
__constant__ int *devChildKernelCount;

int debugInit() {
	void * devPtr;
	cudaError_t status;
	if(status = cudaMalloc(&devPtr, sizeof(int))) {
		errorCuda(status);
		return -1;
	}

	if(status = cudaMemset(devPtr, 0, sizeof(int))) {
		errorCuda(status);
		return -1;
	}

	if(status = cudaMemcpyToSymbol(devChildKernelCount, &devPtr, sizeof(int *))) {
		errorCuda(status);
		return -1;
	}

	return 0;
}

__device__ void kernelCountInc() {

	atomicAdd(devChildKernelCount, 1);
}

int printChildKernelCount() {

	void * devPtr;
	cudaError_t status;
	if(status = cudaMemcpyFromSymbol(&devPtr, devChildKernelCount, sizeof(int *))) {
		errorCuda(status);
		return -1;
	}

	int childKernelCount;
	if(status = cudaMemcpy(&childKernelCount, devPtr, sizeof(int), cudaMemcpyDeviceToHost)) {
		errorCuda(status);
		return -1;
	}

	std::cout << "Child Kernel Count " << childKernelCount << "\n";

	return 0;
}

#endif

}
}

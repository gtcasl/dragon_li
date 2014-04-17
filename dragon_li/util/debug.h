#pragma once

#include <hydrazine/interface/debug.h>

#define errorMsg(x) \
	std::cout << "(" << hydrazine::_debugTime() << ") " \
		<< hydrazine::_debugFile( __FILE__, __LINE__ ) \
		<< " Error: " << x << "\n";

#define errorCuda(x) errorMsg(cudaGetErrorString(x))

#ifndef NDEBUG
	#define reportDevice(...) \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL) { \
			printf("(Device Debug): "); \
			printf(__VA_ARGS__); \
			printf("\n"); \
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
	#define errorCudaDevice()
#endif

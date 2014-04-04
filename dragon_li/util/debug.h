#pragma once

#include <hydrazine/interface/debug.h>

#define errorMsg(x) \
	std::cout << "(" << hydrazine::_debugTime() << ") " \
		<< hydrazine::_debugFile( __FILE__, __LINE__ ) \
		<< " Error: " << x << "\n";

#define errorCuda(x) errorMsg(cudaGetErrorString(x))

#ifndef NDEBUG
	#define reportDevice \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL) \
			printf
#else
	#define reportDevice
#endif


#if !defined(NDEBUG) && REPORT_BASE >= REPORT_ERROR_LEVEL
	#define errorCudaDevice() { \
		cudaError_t retVal = cudaGetLastError(); \
		if(retVal) \
			printf(cudaGetErrorString(x));
#else
	#define errorCudaDevice()
#endif

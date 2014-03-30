#pragma once

#include <hydrazine/interface/debug.h>
#include <dragon_li/util/graphCsrDevice.h>
#include <dragon_li/util/userConfig.h>
#include <dragon_li/util/ctaOutputAssignment.h>

#undef REPORT_BASE
#define REPORT_BASE 1

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsBase {

protected:
	typedef typename Settings::Types Types;
	typedef typename Settings::VertexIdType VertexIdType;
	typedef typename Settings::EdgeWeightType EdgeWeightType;
	typedef typename Settings::SizeType SizeType;
	typedef typename dragon_li::util::GraphCsrDevice<Types> GraphCsrDevice;
	typedef typename dragon_li::util::CtaOutputAssignment<SizeType> CtaOutputAssignment;

public:

	class UserConfig : public dragon_li::util::UserConfig {
	public:
		double frontierScaleFactor;

		UserConfig(
			bool _verbose,
			bool _veryVerbose,
			double _frontierScaleFactor) :
				dragon_li::util::UserConfig(_verbose, _veryVerbose),
				frontierScaleFactor(_frontierScaleFactor) {}
	};


	//Graph CSR information
	SizeType vertexCount;
	SizeType edgeCount;
	VertexIdType * devColumnIndices;
	SizeType * devRowOffsets;

	//Frontiers for bfs
	SizeType maxFrontierSize;
	SizeType frontierSize;
	SizeType * devFrontierSize;
	VertexIdType * devFrontierIn;
	VertexIdType * devFrontierOut;

	//Iteration count
	SizeType iteration;
	SizeType * devIteration;

	//Cta Output Assignemtn
	CtaOutputAssignment ctaOutputAssignment;

	BfsBase() : 
		vertexCount(0),
		edgeCount(0),
		devColumnIndices(NULL),
		devRowOffsets(NULL),
		maxFrontierSize(0),
		frontierSize(0),
		devFrontierSize(NULL),
		devFrontierIn(NULL),
		devFrontierOut(NULL),
		iteration(0),
		devIteration(NULL) {}

	virtual int search() = 0;

	virtual int setup(
					GraphCsrDevice &graphCsrDevice,
					UserConfig &userConfig) {
		return setup(
				graphCsrDevice.vertexCount,
				graphCsrDevice.edgeCount,
				graphCsrDevice.devColumnIndices,
				graphCsrDevice.devRowOffsets,
				userConfig.frontierScaleFactor
			);
	}

	virtual int setup(
			SizeType _vertexCount,
			SizeType _edgeCount,
			VertexIdType * _devColumnIndices,
			SizeType * _devRowOffsets,
			double frontierScaleFactor
		) {
		
			if(!_vertexCount || !_edgeCount
				|| !_devColumnIndices
				|| !_devRowOffsets) {
				errorMsg("Invalid parameters when setting up bfs base!");
				return -1;
			}

			vertexCount = _vertexCount;
			edgeCount = _edgeCount;
			devColumnIndices = _devColumnIndices;
			devRowOffsets = _devRowOffsets;

			report("frontierSF " << frontierScaleFactor);
			maxFrontierSize = (SizeType)((double)edgeCount * frontierScaleFactor);

			cudaError_t retVal;
			if(retVal = cudaMalloc(&devFrontierSize, sizeof(SizeType))) {
				errorCuda(retVal);
				return -1;
			}
			frontierSize = 1; //always start with one vertex in frontier
			if(retVal = cudaMemcpy(devFrontierSize, &frontierSize, sizeof(SizeType),
							cudaMemcpyHostToDevice)) {
				errorCuda(retVal);
				return -1;
			}


			report("MaxFrontierSize " << maxFrontierSize);
			if(retVal = cudaMalloc(&devFrontierIn, maxFrontierSize * sizeof(VertexIdType))) {
				errorCuda(retVal);
				return -1;
			}
			VertexIdType startVertexId = 0; //always expand from id 0
			if(retVal = cudaMemcpy(devFrontierIn, &startVertexId, sizeof(VertexIdType),
							cudaMemcpyHostToDevice)) {
				errorCuda(retVal);
				return -1;
			}

			if(retVal = cudaMalloc(&devFrontierOut, maxFrontierSize * sizeof(VertexIdType))) {
				errorCuda(retVal);
				return -1;
			}

			if(retVal = cudaMalloc(&devIteration, sizeof(SizeType))) {
				errorCuda(retVal);
				return -1;
			}

			if(ctaOutputAssignment.setup() != 0)
				return -1;

			return 0;
		}

	virtual int displayIteration(bool veryVerbose = false) {
		cudaError_t retVal;
		if(retVal = cudaMemcpy(&iteration, devIteration, sizeof(SizeType), 
						cudaMemcpyDeviceToHost)) {
			errorCuda(retVal);
			return -1;
		}
		if(retVal = cudaMemcpy(&frontierSize, devFrontierSize, sizeof(SizeType), 
						cudaMemcpyDeviceToHost)) {
			errorCuda(retVal);
			return -1;
		}

		std::cout << "Iteration " << iteration <<": frontier size "
			<< frontierSize << "\n";

		if(veryVerbose) {
		
			std::vector< VertexIdType > frontier(frontierSize);
		
			if(retVal = cudaMemcpy((void *)(frontier.data()), devFrontierSize, 
							frontierSize * sizeof(VertexIdType), 
							cudaMemcpyDeviceToHost)) {
				errorCuda(retVal);
				return -1;
			}

			std::cout << "Frontier: ";
			for(SizeType i = 0; i < frontierSize; i++) {
				std::cout << frontier[i] << ", ";
			}
			std::cout << "\n";

		}
		return 0;
	}


	virtual int finish() { return 0;}

	virtual int displayResult() { return 0;}	

};

}
}

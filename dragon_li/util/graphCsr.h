#pragma once

#include <vector>
#include <list>
#include <iostream>

#include <hydrazine/interface/debug.h>

#include <dragon_li/util/graphFile.h>
#include <dragon_li/util/types.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace util {

template < typename Types >
class GraphCsr {

	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

	typedef typename GraphFileVertexData< Types >::GraphFileEdgeData GraphFileEdgeData;

public:
	SizeType vertexCount;
	SizeType edgeCount;

	std::vector< VertexIdType > columnIndices;
	std::vector< EdgeWeightType > columnWeights;
	std::vector< SizeType > rowOffsets;

	GraphCsr();

	int buildFromGRFile(const char * fileName);

	int displayCsr();

};

template < typename Types >
GraphCsr< Types >::GraphCsr():
	vertexCount(0), edgeCount(0) {
}

template< typename Types >
int GraphCsr< Types >::buildFromGRFile(const char *fileName) {
	
	GraphFileGR< Types > graphFileGR;

	if(graphFileGR.build(fileName))
		return -1;

	vertexCount = graphFileGR.vertexCount;
	edgeCount = graphFileGR.edgeCount;

	columnIndices.resize(edgeCount);
	columnWeights.resize(edgeCount);
	rowOffsets.resize(vertexCount + 1);

	for(size_t i = 0; i < vertexCount; i++) {
		if(i == 0)
			rowOffsets[0] = 0;
		else {
			rowOffsets[i] = 
				rowOffsets[i - 1] + graphFileGR.vertices[i - 1].degree;
		}

		std::list< GraphFileEdgeData > &edges = 
			graphFileGR.vertices[i].edges;

		size_t startId = rowOffsets[i];
		for(typename std::list< GraphFileEdgeData >::iterator 
				edge = edges.begin(); edge != edges.end(); edge++) {
			assertM(edge->fromVertex == i, "from vertex " << edge->fromVertex
				<< " does not match vertexId " << i);

			columnIndices[startId] = edge->toVertex;
			columnWeights[startId++] = edge->weight;
		}

	}

	rowOffsets[vertexCount] = rowOffsets[vertexCount - 1] + 
		graphFileGR.vertices[vertexCount - 1].degree;

	return 0;
}

template< typename Types >
int GraphCsr< Types >::displayCsr() {
	std::cout << "CSR Graph: vertex count " << vertexCount << ", edge count " << edgeCount << "\n";
	for (size_t vertex = 0; vertex < vertexCount; vertex++) {
		std::cout << vertex << ": ";
		for (size_t edge = rowOffsets[vertex]; edge < rowOffsets[vertex + 1]; edge++) {
			std::cout << columnIndices[edge] << 
			"(" << columnWeights[edge] << ")" << ", ";
		}
		std::cout << "total " << rowOffsets[vertex + 1] - rowOffsets[vertex] << "\n";
	}

	return 0;
}


}
}

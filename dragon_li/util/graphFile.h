#pragma once

#include <list>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>

#include <hydrazine/interface/debug.h>

#include <dragon_li/util/debug.h>
#include <dragon_li/util/types.h>

#undef REPORT_BASE
#define REPORT_BASE 0

namespace dragon_li {
namespace util {


template<typename Types>
class GraphFileVertexData {
	
	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

public:
	class GraphFileEdgeData {
	public:
		VertexIdType fromVertex;
		VertexIdType toVertex;
		EdgeWeightType weight;
	
		GraphFileEdgeData() {}
		GraphFileEdgeData(VertexIdType from, VertexIdType to, EdgeWeightType w):
			fromVertex(from), toVertex(to), weight(w) {}
	};

	VertexIdType vertexId;
	SizeType degree; //count of outgoding edges

	//Edge list
	std::list< GraphFileEdgeData > edges;

	GraphFileVertexData() : vertexId(-1), degree(0) {}
	GraphFileVertexData(VertexIdType id, SizeType d = 0) : vertexId(id), degree(d) {}
//	GraphFileVertexData & operator= (GraphFileVertexData && vertex) {
//		if (this == &vertex) return *this;
//		std::swap(degree, vertex.degree);
//		std::swap(vertexId, vertex.vertexId);
//		std::swap(edges, vertex.edges);
//	}
	GraphFileVertexData & operator= (const GraphFileVertexData & vertex) {
		if (this == &vertex) return *this;
		degree = vertex.degree;
		vertexId = vertex.vertexId;
		edges = vertex.edges;
		return *this;
	}

	GraphFileEdgeData & insertEdge(
		VertexIdType to, EdgeWeightType w) {
		edges.push_back(GraphFileEdgeData(vertexId, to, w)); 
		degree++;
		return edges.back();
	}
	GraphFileEdgeData & insertEdge(
		GraphFileEdgeData & edge) {
		edges.push_back(edge);
		degree++;
		return edges.back();
	}
};

template<typename Types> 
class GraphFileGR {
	
	typedef typename Types::VertexIdType VertexIdType;
	typedef typename Types::EdgeWeightType EdgeWeightType;
	typedef typename Types::SizeType SizeType;

public:
	SizeType vertexCount;
	SizeType edgeCount;

	std::vector< GraphFileVertexData< Types > > vertices;
	
	GraphFileGR();

	int build(const char * fileName);

};


template< typename Types >
GraphFileGR< Types >::GraphFileGR() :
	vertexCount(0), edgeCount(0) {
}

template< typename Types >
int GraphFileGR< Types >::build(const char * fileName) {

	std::ifstream grFile(fileName);
	if(!grFile.is_open()) {
		errorMsg("Error opening file " << fileName);
		return -1;
	}

	char keyWord;
	char tmpFileBuf[256];

	grFile >> keyWord;

	while(!grFile.fail()) {

		if( keyWord == 'p') { 
		
			//Problem line, format: p sp total_vertex_count total_edge_count

			grFile >> tmpFileBuf;

			//p followed by sp
			if(!std::strcmp(tmpFileBuf, "sp")) {

				//get vertex and edge count
				grFile >> vertexCount;
				grFile >> edgeCount;

				vertices.resize(vertexCount);

				//initialize vertex data
				for(SizeType i = 0; i < vertexCount; i++)
					vertices[i] = GraphFileVertexData< Types >(i);
			}
			else{
				errorMsg("Error GR File format for " << fileName);
				grFile.close();
				return -1;
			}

		}
		else if( keyWord == 'a') { //Arc or edge description line
			//format: a from_vertex to_vertex edge_weight

			SizeType fromVertexId, toVertexId;
			EdgeWeightType weight;

			//get edge
			grFile >> fromVertexId >> toVertexId >> weight;

			//GR File always start vertex ID from 1
			fromVertexId--;
			toVertexId--;

			//check boundary
			if(fromVertexId >= vertexCount) {
				errorMsg("VertexId " << fromVertexId << " exceeds limit");
				grFile.close();
				return -1;
			}
			if(toVertexId >= vertexCount) {
				errorMsg("VertexId " << toVertexId << " exceeds limit");
				grFile.close();
				return -1;
			}

			//insert edge to vertex data
			vertices[fromVertexId].insertEdge(toVertexId, weight);

		}
		else if( keyWord != 'c') { //not comment line, then unknown keyword
			errorMsg("Error GR File format for " << fileName);
			grFile.close();
			return -1;
		}

		grFile.getline(tmpFileBuf, 256); //skip to next line

		grFile >> keyWord;

	}

	grFile.close();
	return 0;

}
     	
     	
}    	
}    	
     	

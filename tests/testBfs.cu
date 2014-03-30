#include <string>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/debug.h>

#include <dragon_li/util/types.h>
#include <dragon_li/util/settings.h>
#include <dragon_li/util/graphCsr.h>
#include <dragon_li/util/graphCsrDevice.h>

#include <dragon_li/bfs/bfsReg.h>


#undef REPORT_BASE
#define REPORT_BASE 1


int main(int argc, char **argv) {
	
	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("Dragon Li BFS");

	std::string inputGraphFile;
	parser.parse("-g", "--graph", inputGraphFile, "graphs/sample.gr", "Input graph"); 

	std::string graphFormat;
	parser.parse("-f", "--format", graphFormat, "gr", "Input graph format, default to 'gr'");

	bool displayGraph;
	parser.parse("", "--display", displayGraph, false, "Display input graph");

	bool verbose;
	parser.parse("-v", "--verbose", verbose, false, "Verbose, display information");

	bool veryVerbose;
	parser.parse("", "--verbose2", veryVerbose, false, "Very verbose, display extended information");

	double frontierScaleFactor;
	parser.parse("", "--sf", frontierScaleFactor, 1.0, "Frontier scale factor, default 1.0");

	parser.parse();

	typedef dragon_li::util::Types< int,	//VertexIdType
							int,	//EdgeWeightType
							int		//SizeType
							> Types;
	dragon_li::util::GraphCsr< Types > graph;

	if(!graphFormat.compare("gr")) {

		if(graph.buildFromGRFile(inputGraphFile.c_str()))
			return -1;
	}
	else {
		errorMsg("Unrecoginized graph format");
		return -1;
	}

	if(displayGraph)
		graph.displayCsr();

	dragon_li::util::GraphCsrDevice< Types > graphDev;
	if(graphDev.setup(graph))
		return -1;

	typedef dragon_li::util::Settings< 
				Types,		//types
				256, 		//THREADS
				260			//CTAS
				> Settings;

	dragon_li::bfs::BfsReg< Settings > bfsReg;
	dragon_li::bfs::BfsReg< Settings >::UserConfig bfsRegConfig(
													verbose,
													veryVerbose,
													frontierScaleFactor);

	if(bfsReg.setup(graphDev, bfsRegConfig))
		return -1;

	if(bfsReg.search())
		return -1;

	return 0;
}

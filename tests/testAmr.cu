#include <string>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/debug.h>

#include <dragon_li/amr/types.h>
#include <dragon_li/amr/settings.h>
#include <dragon_li/amr/amrReg.h>
//#include <dragon_li/amr/amrCdp.h>
//#include <dragon_li/amr/amrCpu.h>

#undef REPORT_BASE
#define REPORT_BASE 0


int main(int argc, char **argv) {
	
	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("Dragon Li AMR");

	bool verbose;
	parser.parse("-v", "--v1", verbose, false, "Verbose, display information");

	bool veryVerbose;
	parser.parse("", "--v2", veryVerbose, false, "Very verbose, display extended information");

	bool verify;
	parser.parse("-e", "--verify", verify, false, "Verify results against CPU implementation");

	bool cdp; //use CDP
	parser.parse("", "--cdp", cdp, false, "Use Cuda Dynamic Parallelism");

	int maxGridDataSize;
	parser.parse("", "--maxGridDataSize", maxGridDataSize, 1024*1024, "Max Grid Size (cell count)");

	int maxRefineLevel;
	parser.parse("", "--maxRefineLevel", maxRefineLevel, 32, "Max level to refine the grid"); 
	
	parser.parse();

	//Basic Types and Settings
	typedef dragon_li::util::Types<
							int 			//SizeType
							> _Types;
	typedef dragon_li::util::Settings< 
				_Types,						//types
				256, 						//THREADS
				104,						//CTAS
				5,							//CDP_THREADS_BITS
				32							//CDP_THRESHOLD
				> _Settings;



	typedef dragon_li::amr::Types<
							_Types, 		//Basic Types
							int				//DataType
							> Types;
	typedef dragon_li::amr::Settings<
				_Settings, 					//Basic Settings
				Types,						//AMR Types
				32,							//GRID_REFINE_SIZE
				32,							//GRID_REFINE_THRESHOLD
				1024						//MAX_GRID_VALUE
				> Settings;


	if(!cdp) {
		dragon_li::amr::AmrReg< Settings > amrReg;
		dragon_li::amr::AmrReg< Settings >::UserConfig amrRegConfig(
														verbose,
														veryVerbose,
														maxGridDataSize,
														maxRefineLevel);
	
		if(amrReg.setup(amrRegConfig))
			return -1;
	
		if(amrReg.refine())
			return -1;
	
//		if(verify) {
//			dragon_li::bfs::BfsCpu<Types>::bfsCpu(graph);
//			if(!bfsReg.verifyResult(dragon_li::bfs::BfsCpu<Types>::cpuSearchDistance)) {
//				std::cout << "Verify correct!\n";
//			}
//			else {
//				std::cout << "Incorrect!\n";
//			}
//		}
	
		if(amrReg.displayResult())
			return -1;
	}
//	else {
//		dragon_li::bfs::BfsCdp< Settings > bfsCdp;
//		dragon_li::bfs::BfsCdp< Settings >::UserConfig bfsCdpConfig(
//														verbose,
//														veryVerbose,
//														frontierScaleFactor);
//	
//		if(bfsCdp.setup(graphDev, bfsCdpConfig))
//			return -1;
//	
//		if(bfsCdp.search())
//			return -1;
//	
//		if(verify) {
//			dragon_li::bfs::BfsCpu<Types>::bfsCpu(graph);
//			if(!bfsCdp.verifyResult(dragon_li::bfs::BfsCpu<Types>::cpuSearchDistance)) {
//				std::cout << "Verify correct!\n";
//			}
//			else {
//				std::cout << "Incorrect!\n";
//			}
//		}
//	
//		if(bfsCdp.displayResult())
//			return -1;
//	}


	return 0;
}

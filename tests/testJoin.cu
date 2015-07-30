#include <string>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/debug.h>

#include <dragon_li/join/types.h>
#include <dragon_li/join/settings.h>
#include <dragon_li/join/joinReg.h>
//#include <dragon_li/join/joinCdp.h>
#include <dragon_li/join/joinCpu.h>

#undef REPORT_BASE
#define REPORT_BASE 0


int main(int argc, char **argv) {
	
	//Basic Types and Settings
	typedef dragon_li::util::Types<
							int 			//SizeType
							> _Types;
	typedef dragon_li::util::Settings< 
				_Types,						//types
				128, 						//THREADS
				240,						//CTAS
				5,							//CDP_THREADS_BITS
				32							//CDP_THRESHOLD
				> _Settings;



	typedef dragon_li::join::Types<
							_Types, 		//Basic Types
							int				//DataType
							> Types;
	typedef dragon_li::join::Settings<
				_Settings, 					//Basic Settings
				Types,						//Join Types
				300,						//Join Scale Factor
				100							//Join block scalefactor
				> Settings;

	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("Dragon Li JOIN");

	bool verbose;
	parser.parse("-v", "--v1", verbose, false, "Verbose, display information");

	bool veryVerbose;
	parser.parse("", "--v2", veryVerbose, false, "Very verbose, display extended information");

	bool verify;
	parser.parse("-e", "--verify", verify, false, "Verify results against CPU implementation");

	bool cdp; //use CDP
	parser.parse("", "--cdp", cdp, false, "Use Cuda Dynamic Parallelism");


	Settings::SizeType inputLeftCount;
	parser.parse("-l", "--leftCnt", inputLeftCount, 1024, "Left input element count");

	Settings::SizeType inputRightCount;
	parser.parse("-r", "--righCnt", inputRightCount, 1024, "Right input element count");

	parser.parse();


	if(!cdp) {
		dragon_li::join::JoinReg< Settings > joinReg;
		dragon_li::join::JoinReg< Settings >::UserConfig joinRegConfig(
														verbose,
														veryVerbose
														);
	
		dragon_li::join::JoinData< Types > joinData;
		joinData.generateRandomData(inputLeftCount, inputRightCount);

		if(joinReg.setup(joinData, joinRegConfig))
			return -1;
	
		if(joinReg.join())
			return -1;
	
		if(verify) {
			dragon_li::join::JoinCpu<Types>::joinCpu(joinData);
			if(!joinReg.verifyResult(
				dragon_li::join::JoinCpu<Types>::cpuJoinLeftIndices, 
				dragon_li::join::JoinCpu<Types>::cpuJoinRightIndices,
				joinData)) {
				std::cout << "Verify correct!\n";
			}
			else {
				std::cout << "Incorrect!\n";
			}
		}
	
		if(joinReg.displayResult())
			return -1;

		if(joinReg.finish())
			return -1;
	}
//	else {
//		dragon_li::join::JoinCdp< Settings > joinCdp;
//		dragon_li::join::JoinCdp< Settings >::UserConfig joinCdpConfig(
//														verbose,
//														veryVerbose,
//														maxGridDataSize,
//														maxRefineLevel,
//														maxGridValue,
//														gridRefineThreshold
//														);
//	
//
//		if(joinCdp.setup(joinCdpConfig))
//			return -1;
//	
//		if(joinCdp.refine())
//			return -1;
//	
////		if(verify) {
////			dragon_li::join::JoinCpu<Types>::joinCpu(graph);
////			if(!joinCdp.verifyResult(dragon_li::join::JoinCpu<Types>::cpuSearchDistance)) {
////				std::cout << "Verify correct!\n";
////			}
////			else {
////				std::cout << "Incorrect!\n";
////			}
////		}
//	
//		if(joinCdp.displayResult())
//			return -1;
//	}


	return 0;
}

#pragma once

#include <dragon_li/bfs/bfsReg.h>

namespace dragon_li {
namespace bfs {

template< typename Settings >
class BfsCdp : public BfsReg< Settings > {

public:

	BfsCdp() : BfsReg< Settings >() {}

	int expand() {

		bfsCdpExpandKernel< Settings >
			<<< CTAS, THREADS >>> (
		);

	}
}

}
}

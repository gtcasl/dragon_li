#pragma once

#include <dragon_li/util/types.h>

namespace dragon_li {
namespace util {

template< 
			typename _Types,
			int _THREADS,
			int _CTAS
		>
class Settings {

public:
	typedef _Types Types;
	typedef typename _Types::VertexIdType VertexIdType;
	typedef typename _Types::EdgeWeightType EdgeWeightType;
	typedef typename _Types::SizeType SizeType;

	static const SizeType THREADS = _THREADS;
	static const SizeType CTAS = _CTAS;

};

}
}

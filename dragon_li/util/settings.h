#pragma once

#include <dragon_li/util/types.h>

namespace dragon_li {
namespace util {

template< 
			typename _Types,
			int _THREADS,
			int _CTAS,
			int _MASK_BITS
		>
class Settings {

public:
	typedef _Types Types;
	typedef typename _Types::VertexIdType VertexIdType;
	typedef typename _Types::EdgeWeightType EdgeWeightType;
	typedef typename _Types::SizeType SizeType;
	typedef typename _Types::MaskType MaskType;

	static const SizeType THREADS = _THREADS;
	static const SizeType CTAS = _CTAS;
	static const SizeType MASK_BITS = _MASK_BITS;
	static const SizeType MASK_SIZE = 1 << MASK_BITS;
	static const SizeType MASK_MASK = MASK_SIZE - 1;

};

}
}

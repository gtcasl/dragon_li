#pragma once

namespace dragon_li {
namespace util {


template<
	typename _VertexIdType,
	typename _EdgeWeightType,
	typename _SizeType,
	typename _MaskType
	>
class Types {

public:
	typedef _VertexIdType VertexIdType;
	typedef _EdgeWeightType EdgeWeightType;
	typedef _SizeType SizeType;
	typedef _MaskType MaskType;
};

}
}

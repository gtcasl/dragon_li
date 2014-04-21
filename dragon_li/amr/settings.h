#pragma once

#include <dragon_li/util/settings.h>

namespace dragon_li {
namespace amr {

template< 
			typename _Settings,
			typename _Types,
			typename _Types::SizeType _GRID_REFINE_SIZE,
			typename _Types::DataType _GRID_REFINE_THRESHOLD,
			typename _Types::DataType _MAX_GRID_VALUE
		>
class Settings : public _Settings{

public:
	typedef _Types Types;
	typedef typename Types::DataType DataType;
	typedef typename Types::SizeType SizeType;

	static const SizeType GRID_REFINE_SIZE = _GRID_REFINE_SIZE;
	static const DataType GRID_REFINE_THRESHOLD = _GRID_REFINE_THRESHOLD;
	static const DataType MAX_GRID_VALUE = _MAX_GRID_VALUE;
};

}
}

#pragma once

#include <dragon_li/util/settings.h>

namespace dragon_li {
namespace join {

template< 
			typename _Settings,
			typename _Types
		>
class Settings : public _Settings{

public:
	typedef _Types Types;
	typedef typename Types::DataType DataType;
	typedef typename Types::SizeType SizeType;
};

}
}

#pragma once

#include <cuda.h>
#include <VMUtils/modules.hpp>

namespace cufx
{
VM_BEGIN_MODULE( drv )

struct Init
{
	Init()
	{
		static auto _ = [&] {
			cuInit( 0 );
			return 0;
		}();
	}
};

VM_END_MODULE()

}  // namespace cufx

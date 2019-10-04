#pragma once

#include <cuda.h>
#include "internal/init.hpp"
#include "../device_id.hpp"

namespace cufx
{
VM_BEGIN_MODULE( drv )

VM_EXPORT
{
	struct Context
	{
		Context( unsigned int flags, DeviceId const &device = DeviceId{} )
		{
			static Init init;
			CUdevice dev = 0;
			cuDeviceGet( &dev, device.id() );
			cuCtxCreate( &_, flags, dev );
		}
		operator CUcontext() const { return _; }

	private:
		CUcontext _;
	};
}

VM_END_MODULE()

}  // namespace cufx

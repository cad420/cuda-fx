#pragma once

#include <cuda_runtime.h>
#include <VMUtils/modules.hpp>

VM_BEGIN_MODULE( cufx )

VM_EXPORT{

#define CUFX_DEVICE_CODE ( defined( __CUDA_ARCH__ ) && defined( __CUDACC__ ) )

}

VM_END_MODULE()

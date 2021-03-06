#pragma once

#include <cuda_runtime.h>
#include <VMUtils/modules.hpp>

#include "internal/attribute.hpp"
#include "internal/format.hpp"

VM_BEGIN_MODULE( cufx )

using namespace std;

VM_EXPORT
{
	struct Extent
	{
		CUFX_DEFINE_ATTRIBUTE( size_t, width );
		CUFX_DEFINE_ATTRIBUTE( size_t, height );
		CUFX_DEFINE_ATTRIBUTE( size_t, depth );

	public:
		__host__ __device__ size_t size() const { return width * height * depth; }
		__host__ __device__ cudaExtent get() const { return cudaExtent{ width, height, depth }; }
	};
}

VM_END_MODULE()

#define CUFX_DEFINE_VECTOR1234_FMT( T )      \
	CUFX_DEFINE_VECTOR1_FMT( T##1, x )       \
	CUFX_DEFINE_VECTOR2_FMT( T##2, x, y )    \
	CUFX_DEFINE_VECTOR3_FMT( T##3, x, y, z ) \
	CUFX_DEFINE_VECTOR4_FMT( T##4, x, y, z, w )

CUFX_DEFINE_VECTOR1234_FMT( char )
CUFX_DEFINE_VECTOR1234_FMT( uchar )
CUFX_DEFINE_VECTOR1234_FMT( short )
CUFX_DEFINE_VECTOR1234_FMT( ushort )
CUFX_DEFINE_VECTOR1234_FMT( int )
CUFX_DEFINE_VECTOR1234_FMT( uint )
CUFX_DEFINE_VECTOR1234_FMT( long )
CUFX_DEFINE_VECTOR1234_FMT( ulong )
CUFX_DEFINE_VECTOR1234_FMT( longlong )
CUFX_DEFINE_VECTOR1234_FMT( ulonglong )
CUFX_DEFINE_VECTOR1234_FMT( float )
CUFX_DEFINE_VECTOR1234_FMT( double )
CUFX_DEFINE_VECTOR3_FMT( dim3, x, y, z )
CUFX_DEFINE_VECTOR3_FMT( ::cufx::Extent, width, height, depth )
CUFX_DEFINE_VECTOR3_FMT( cudaExtent, width, height, depth )

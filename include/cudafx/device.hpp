#pragma once

#include <vector>
#include <VMUtils/modules.hpp>

#include "device_id.hpp"
#include "memory.hpp"
#include "image.hpp"
#include "array.hpp"

VM_BEGIN_MODULE( cufx )

using namespace std;

VM_EXPORT
{
	struct Device : DeviceId
	{
		static vector<Device> scan()
		{
			int n;
			cudaGetDeviceCount( &n );
			vector<Device> _;
			for ( int i = 0; i != n; ++i ) {
				Device e( i );
				_.emplace_back( std::move( e ) );
			}
			return _;
		}

	public:
		GlobalMemory alloc_global( size_t size ) const
		{
			return GlobalMemory( size, *this );
		}
		template <typename E, size_t N, typename... Args>
		ArrayND<E, N> alloc_arraynd( Args &&... args ) const
		{
			return ArrayND<E, N>( std::forward<Args>( args )..., *this );
		}
		template <typename Pixel>
		std::pair<GlobalMemory, MemoryView2D<Pixel>> alloc_image_swap( Image<Pixel> const &img ) const
		{
			auto img_view = img.view();
			cufx::GlobalMemory mem( img_view.width() * img_view.height() * sizeof( Pixel ), *this );
			auto view_info = cufx::MemoryView2DInfo{}
							   .set_stride( img_view.width() * sizeof( Pixel ) )
							   .set_width( img_view.width() )
							   .set_height( img_view.height() );
			auto view = mem.view_2d<Pixel>( view_info );
			return make_pair( mem, view );
		}

	private:
		Device( int _ ) :
		  DeviceId( _ ) {}
	};
}

VM_END_MODULE()

#pragma once

#include <vector>
#include <VMUtils/modules.hpp>
#include <VMUtils/option.hpp>

#include "device_id.hpp"
#include "memory.hpp"
#include "image.hpp"
#include "array.hpp"
#include "driver/context.hpp"

VM_BEGIN_MODULE( cufx )

using namespace std;

VM_EXPORT
{
	struct WorkerThread;
	
	struct Device : DeviceId
	{
		static vector<Device> scan()
		{
			int n = 0;
			cudaGetDeviceCount( &n );
			vector<Device> _;
			for ( int i = 0; i != n; ++i ) {
				Device e( i );
				_.emplace_back( std::move( e ) );
			}
			return _;
		}

		static Device get_current()
		{
			int n = 0;
			cudaGetDevice( &n );
			return Device( n );
		}

		static vm::Option<Device> get_default()
		{
			int n = 0;
			cudaGetDeviceCount( &n );
			cudaGetLastError();
			if ( n == 0 ) return vm::None{};
			return Device( 0 );
		}

	public:
		drv::Context create_context( unsigned int flags ) const
		{
			return drv::Context( flags, *this );
		}
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
		
		friend struct WorkerThread;
	};

	struct WorkerThread : std::thread
	{
		WorkerThread( std::function<void()> const &runner,
					  vm::Option<Device> const &device ) :
			std::thread( [runner, device, this] {
					if ( device.has_value() ) {
						this->lk = device.value().lock();
					}
					runner();
				} )
		{
		}

		WorkerThread( std::function<void()> const &runner,
					  int device ) :
			WorkerThread( runner, Device( device ) )
		{
		}
	private:
		vm::Option<Device::Lock> lk;
	};
}

VM_END_MODULE()

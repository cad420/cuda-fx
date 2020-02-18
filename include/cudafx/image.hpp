#pragma once

#include <iostream>
#include <stb/stb_image_write.h>
#include <VMUtils/modules.hpp>

#include "memory.hpp"
#include "transfer.hpp"
#include "internal/attribute.hpp"

VM_BEGIN_MODULE( cufx )

using namespace std;

VM_EXPORT
{
	struct Rect
	{
		CUFX_DEFINE_ATTRIBUTE( size_t, x0 ) = 0;
		CUFX_DEFINE_ATTRIBUTE( size_t, y0 ) = 0;
		CUFX_DEFINE_ATTRIBUTE( size_t, x1 ) = 0;
		CUFX_DEFINE_ATTRIBUTE( size_t, y1 ) = 0;

	public:
		size_t width() const { return x1 - x0; }
		size_t height() const { return y1 - y0; }
	};

	template <typename Pixel>
	struct Image;

	template <typename Pixel = uchar4>
	struct ImageView final
	{
		__host__ __device__ Pixel &at_host( size_t x, size_t y ) const { return host_mem.at( x, y ); }
		__host__ __device__ Pixel &at_device( size_t x, size_t y ) const { return device_mem.at( x, y ); }
		__host__ __device__ size_t width() const { return host_mem.width(); }
		__host__ __device__ size_t height() const { return host_mem.height(); }

	public:
		ImageView with_device_memory( MemoryView2D<Pixel> const &memory ) const
		{
			auto _ = *this;
			if ( memory.device_id().is_host() ) {
				throw runtime_error( "invalid device memory view" );
			}
			_.device_mem = memory;
			return _;
		}
		Task copy_from_device() const
		{
			return memory_transfer( host_mem, device_mem );
		}
		Task copy_to_device() const
		{
			return memory_transfer( device_mem, host_mem );
		}

	private:
		ImageView( MemoryView2D<Pixel> const &mem ) :
		  host_mem( mem ) {}

	private:
		MemoryView2D<Pixel> host_mem;
		MemoryView2D<Pixel> device_mem;
		friend struct Image<Pixel>;
	};

	struct StdByte4Pixel
	{
		__host__ __device__ void
		  write_to( uchar4 &dst ) const { dst = val; }

	private:
		uchar4 val;
	};

	template <typename Pixel = StdByte4Pixel>
	struct Image final
	{
	private:
		struct Inner : vm::NoCopy, vm::NoMove
		{
			Inner( size_t width, size_t height ) :
			  width( width ), height( height ), pixels( width * height )
			{
			}

			size_t width, height;
			std::vector<Pixel> pixels;
		};

	public:
		Image( size_t width, size_t height )
		{
			_.reset( new Inner( width, height ) );
		}

	public:
		Pixel &at( size_t x, size_t y ) const { return _->pixels[ x + y * _->width ]; }

		ImageView<Pixel> view( Rect const &region ) const
		{
			auto ptr_region = reinterpret_cast<char *>( &at( region.x0, region.y0 ) );
			auto ptr_region_ln1 = reinterpret_cast<char *>( &at( region.x0, region.y0 + 1 ) );
			auto view = MemoryView2DInfo{}
						  .set_stride( ptr_region_ln1 - ptr_region )
						  .set_width( region.width() )
						  .set_height( region.height() );
			auto mem = MemoryView2D<Pixel>( ptr_region, view );
			return ImageView<Pixel>( mem );
		}
		ImageView<Pixel> view() const
		{
			return view( Rect{}.set_x0( 0 ).set_y0( 0 ).set_x1( _->width ).set_y1( _->height ) );
		}

		void dump( ImageView<StdByte4Pixel> const &view ) const
		{
			for ( int i = 0; i != _->height; ++i ) {
				for ( int j = 0; j != _->width; ++j ) {
					at( j, i ).write_to(
					  *reinterpret_cast<uchar4 *>( &view.at_host( j, i ) ) );
				}
			}
		}
		Image<> dump() const
		{
			auto img = Image<>( _->width, _->height );
			dump( img.view() );
			return img;
		}
		void dump( string const &file_name ) const
		{
			auto img = dump();
			stbi_write_png( file_name.c_str(), _->width, _->height, 4,
							reinterpret_cast<unsigned char *>( img._->pixels.data() ), _->width * 4 );
		}

		size_t get_width() const { return _->width; }
		size_t get_height() const { return _->height; }

	private:
		shared_ptr<Inner> _;
		template <typename T>
		friend struct Image;
	};

	template <>
	inline Image<> Image<>::dump() const
	{
		return *this;
	}

	template <>
	inline void Image<>::dump( string const &file_name ) const
	{
		stbi_write_png( file_name.c_str(), _->width, _->height, 4,
						reinterpret_cast<unsigned char *>( _->pixels.data() ), _->width * 4 );
	}
}

VM_END_MODULE()

#pragma once

#include <vector>
#include <VMUtils/modules.hpp>

#include "stream.hpp"
#include "device_id.hpp"
#include "misc.hpp"

#include "internal/attribute.hpp"

VM_BEGIN_MODULE( cufx )

using namespace std;

VM_EXPORT
{
	struct GlobalMemory;
}

template <typename T, size_t N>
struct MemoryViewNDImpl
{
	__host__ __device__ T *data() const { return reinterpret_cast<T *>( _.ptr ); }
	__host__ __device__ explicit operator bool() const { return _.ptr; }
	cudaPitchedPtr get() const { return _; }
	DeviceId device_id() const { return device; }

protected:
	cudaPitchedPtr _ = { 0 };
	DeviceId device = DeviceId{ -1 };
	friend struct __exported__::GlobalMemory;
};

VM_EXPORT
{
	template <typename T, size_t N>
	struct MemoryViewND;

	struct MemoryView2DInfo
	{
		CUFX_DEFINE_ATTRIBUTE( size_t, stride ) = 0;
		CUFX_DEFINE_ATTRIBUTE( size_t, width ) = 0;
		CUFX_DEFINE_ATTRIBUTE( size_t, height ) = 0;
	};

	template <typename T>
	struct MemoryViewND<T, 1> : MemoryViewNDImpl<T, 1>
	{
		__host__ __device__ T *ptr() const
		{
			return reinterpret_cast<T *>( this->_.ptr );
		}
		__host__ __device__ T &at( size_t x ) const
		{
			return reinterpret_cast<T *>( this->_.ptr )[ x ];
		}
		__host__ __device__ size_t size() const { return this->_.xsize; }
		__host__ __device__ MemoryViewND<T, 1> slice( size_t beg, size_t len ) const
		{
			auto view = *this;
			view._ = cudaPitchedPtr{ this->ptr() + beg, 0, len, 0 };
			return view;
		}

	public:
		MemoryViewND() = default;
		MemoryViewND( void *ptr, size_t len )
		{
			this->_ = cudaPitchedPtr{ ptr, 0, len, 0 };
		}
		template <typename U, typename A>
		MemoryViewND( std::vector<U, A> &vec )
		{
			this->_ = cudaPitchedPtr{ vec.data(), 0, vec.size() * sizeof( U ), 0 };
		}
	};

	template <typename T>
	struct MemoryViewND<T, 2> : MemoryViewNDImpl<T, 2>
	{
		__host__ __device__ T *ptr() const
		{
			return reinterpret_cast<T *>( this->_.ptr );
		}
		__host__ __device__ T &at( size_t x, size_t y ) const
		{
			auto ptr = reinterpret_cast<char *>( this->_.ptr );
			auto line = reinterpret_cast<T *>( ptr + y * this->_.pitch );
			return line[ x ];
		}
		__host__ __device__ size_t width() const { return this->_.xsize; }
		__host__ __device__ size_t height() const { return this->_.ysize; }

	public:
		MemoryViewND() = default;
		MemoryViewND( void *ptr, MemoryView2DInfo const &info )
		{
			this->_ = cudaPitchedPtr{ ptr, info.stride, info.width, info.height };
		}
	};

	template <typename T>
	struct MemoryViewND<T, 3> : MemoryViewNDImpl<T, 3>
	{
		__host__ __device__ T &at( size_t x, size_t y, size_t z ) const
		{
			auto ptr = reinterpret_cast<char *>( this->_.ptr ) +
					   this->_.pitch * this->_.ysize * z;
			auto line = reinterpret_cast<T *>( ptr + y * this->_.pitch );
			return line[ x ];
		}
		__host__ __device__ cudaExtent extent() const { return dim.get(); }

	public:
		MemoryViewND() = default;
		MemoryViewND( void *ptr, MemoryView2DInfo const &info, cufx::Extent dim ) :
		  dim( dim )
		{
			this->_ = cudaPitchedPtr{ ptr, info.stride, info.width, info.height };
		}

	private:
		cufx::Extent dim;
	};

	struct GlobalMemory
	{
	private:
		struct Inner : vm::NoCopy, vm::NoMove
		{
			~Inner() { cudaFree( _ ); }

			char *_;
			size_t size;
			DeviceId device;
		};

	public:
		GlobalMemory( size_t size, DeviceId const &device = DeviceId{} )
		{
			auto lock = device.lock();
			cudaMalloc( &_->_, _->size = size );
			_->device = device;
		}

		size_t size() const { return _->size; }
		char *get() const { return _->_; }

	public:
		template <typename T>
		MemoryViewND<T, 1> view_1d( size_t len, size_t offset = 0 ) const
		{
			auto mem = MemoryViewND<T, 1>( _->_ + offset, len );
			static_cast<MemoryViewNDImpl<T, 1> &>( mem ).device = _->device;
			return mem;
		}
		template <typename T>
		MemoryViewND<T, 2> view_2d( MemoryView2DInfo const &info, size_t offset = 0 ) const
		{
			auto mem = MemoryViewND<T, 2>( _->_ + offset, info );
			static_cast<MemoryViewNDImpl<T, 2> &>( mem ).device = _->device;
			return mem;
		}
		template <typename T>
		MemoryViewND<T, 3> view_3d( MemoryView2DInfo const &info, cufx::Extent dim, size_t offset = 0 ) const
		{
			auto mem = MemoryViewND<T, 3>( _->_ + offset, info, dim );
			static_cast<MemoryViewNDImpl<T, 3> &>( mem ).device = _->device;
			return mem;
		}

	private:
		shared_ptr<Inner> _ = make_shared<Inner>();
	};

	template <typename T>
	using MemoryView1D = MemoryViewND<T, 1>;
	template <typename T>
	using MemoryView2D = MemoryViewND<T, 2>;
	template <typename T>
	using MemoryView3D = MemoryViewND<T, 3>;
}

VM_END_MODULE()

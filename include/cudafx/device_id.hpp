#pragma once

#include <VMUtils/concepts.hpp>
#include "common_def.hpp"

VM_BEGIN_MODULE( cufx )

VM_EXPORT
{
	struct DeviceId
	{
	public:
		struct Lock : vm::NoCopy
		{
			Lock( int _ ) :
			  _( _ ) {}
			Lock( Lock &&_ ) :
			  _( _._ )
			{
				_._ = -1;
			}
			Lock &operator=( Lock && ) = delete;
			~Lock()
			{
				if ( _ >= 0 ) { cudaSetDevice( _ ); }
			}

		private:
			int _;
		};

		using Props = cudaDeviceProp;

	public:
		int id() const { return _; }
		bool is_host() const { return _ < 0; }
		bool is_device() const { return _ >= 0; }
		Lock lock() const
		{
			int d = -1;
			if ( _ >= 0 ) {
				cudaGetDevice( &d );
				cudaSetDevice( _ );
			}
			return Lock( d );
		}
		Props props() const
		{
			Props val;
			cudaGetDeviceProperties( &val, _ );
			return val;
		}
		std::size_t free_memory_bytes() const
		{
			std::size_t free, total;
			auto _ = this->lock();
			cudaMemGetInfo( &free, &total );
			return free;
		}
		std::size_t total_memory_bytes() const
		{
			std::size_t free, total;
			auto _ = this->lock();
			cudaMemGetInfo( &free, &total );
			return total;
		}

		explicit DeviceId( int _ = 0 ) :
		  _( _ ) {}

	private:
		int _;
	};
}

VM_END_MODULE()

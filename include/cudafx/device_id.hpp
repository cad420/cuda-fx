#pragma once

#include <cuda_runtime.h>
#include <mutex>

#include <VMUtils/concepts.hpp>
#include <VMUtils/modules.hpp>

VM_BEGIN_MODULE( cufx )

VM_EXPORT
{
	struct DeviceId
	{
	private:
		struct Lock : vm::NoCopy, vm::NoHeap
		{
			Lock( int _ ) :
			  _( _ )
			{
				int d = -1;
				if ( _ >= 0 ) {
					lock().lock();
					cudaGetDevice( &d );
					cudaSetDevice( _ );
				}
			}
			Lock( Lock &&_ ) :
			  _( _._ )
			{
				_._ = -1;
			}
			Lock &operator=( Lock && ) = delete;
			~Lock()
			{
				if ( _ >= 0 ) {
					cudaSetDevice( _ );
					lock().unlock();
				}
			}

		private:
			static std::recursive_mutex &lock()
			{
				std::recursive_mutex _;
				return _;
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
			return Lock( _ );
		}
		Props props() const
		{
			Props val;
			cudaGetDeviceProperties( &val, _ );
			return val;
		}

		explicit DeviceId( int _ = 0 ) :
		  _( _ ) {}

	private:
		int _;
	};
}

VM_END_MODULE()

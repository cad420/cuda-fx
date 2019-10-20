#pragma once

#include <future>
#include <cstdint>
#include <memory>
#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <cuda_runtime.h>

#include <VMUtils/concepts.hpp>
#include <VMUtils/modules.hpp>
#include <VMUtils/fmt.hpp>
// #ifndef __CUDACC__
// #include <future/index.hpp>
// #endif

VM_BEGIN_MODULE( cufx )

using namespace std;

VM_EXPORT
{
	enum class Poll : uint32_t
	{
		Pending = 0,
		Done = 1,
		Error = 2
	};

	inline ostream &operator<<( ostream &os, Poll stat )
	{
		switch ( stat ) {
		case Poll::Pending: return os << "Pending";
		case Poll::Done: return os << "Done";
		case Poll::Error: return os << "Error";
		default: throw runtime_error( "invalid internal state: Poll" );
		}
	}

	inline Poll from_cuda_poll_result( cudaError_t ret )
	{
		switch ( ret ) {
		case cudaSuccess:
			return Poll::Done;
		case cudaErrorNotReady:
			return Poll::Pending;
		default:
			return Poll::Error;
		}
	}

	struct Result
	{
		Result( cudaError_t _ = cudaSuccess ) :
		  _( _ ) {}

		bool ok() const { return _ == cudaSuccess; }
		bool err() const { return !ok(); }
		explicit operator bool() const { return ok(); }
		const char *name() const { return cudaGetErrorName( _ ); }
		const char *message() const { return cudaGetErrorString( _ ); }
		void unwrap() const
		{
			if ( err() ) {
				vm::eprintln( "Result unwrap failed: {}", message() );
				std::abort();
			}
		}

	private:
		cudaError_t _;
	};

	inline ostream &operator<<( ostream &os, Result stat )
	{
		if ( stat.ok() ) {
			return os << "Ok";
		} else {
			return os << "Err: " << stat.message();
		}
	}

	struct Event
	{
	private:
		struct Inner : vm::NoCopy, vm::NoMove
		{
			~Inner() { cudaEventDestroy( _ ); }

			cudaEvent_t _;
		};

	public:
		Event( bool enable_timing = false )
		{
			unsigned flags = cudaEventBlockingSync;
			if ( !enable_timing ) flags |= cudaEventDisableTiming;
			cudaEventCreateWithFlags( &_->_, flags );
		}

		void record() const { cudaEventRecord( _->_ ); }
		Poll poll() const { return from_cuda_poll_result( cudaEventQuery( _->_ ) ); }
		Result wait() const { return cudaEventSynchronize( _->_ ); }

	public:
		static chrono::microseconds elapsed( Event const &a, Event const &b )
		{
			float dt;
			cudaEventElapsedTime( &dt, a._->_, b._->_ );
			return chrono::microseconds( uint64_t( dt * 1000 ) );
		}

	private:
		shared_ptr<Inner> _ = make_shared<Inner>();
	};

	struct Stream
	{
	private:
		struct Inner : vm::NoCopy, vm::NoMove
		{
			~Inner()
			{
				if ( _ != 0 ) cudaStreamDestroy( _ );
			}

			cudaStream_t _ = 0;
			recursive_mutex mtx;
		};

		Stream( nullptr_t ) {}

	public:
		struct Lock : vm::NoCopy, vm::NoHeap
		{
			Lock( Inner &stream ) :
			  stream( stream ),
			  _( stream.mtx )
			{
			}

			cudaStream_t get() const { return stream._; }

		private:
			Inner &stream;
			unique_lock<recursive_mutex> _;
		};

	public:
		Stream() { cudaStreamCreate( &_->_ ); }

		Poll poll() const { return from_cuda_poll_result( cudaStreamQuery( _->_ ) ); }
		Result wait() const { return cudaStreamSynchronize( _->_ ); }
		Lock lock() const { return Lock( *_ ); }

	public:
		static Stream null() { return Stream( nullptr ); }

	private:
		shared_ptr<Inner> _ = make_shared<Inner>();
	};

	struct Task : vm::NoCopy
	{
	private:
// #ifndef __CUDACC__
// 		struct FutureImpl : koi::Future<Result>
// 		{
// 			koi::future::PollState poll() override
// 			{
// 				if ( first ) {
// 					launch();
// 					first = false;
// 				}
// 				switch ( stop.poll() ) {
// 				case Poll::Done: return koi::future::PollState::Ok;
// 				default: return koi::future::PollState::Pruned;
// 				case Poll::Pending: return koi::future::PollState::Pending;
// 				}
// 			}
// 			Result get() override
// 			{
// 				return stop.wait();
// 			}

// 		private:
// 			FutureImpl( Stream const &stream, vector<function<void( cudaStream_t )>> _ ) :
// 			  stream( stream ),
// 			  _( std::move( _ ) )
// 			{
// 			}

// 			void launch()
// 			{
// 				auto lock = stream.lock();
// 				start.record();
// 				for ( auto &e : this->_ ) e( lock.get() );
// 				stop.record();
// 			}

// 		private:
// 			Event start, stop;
// 			Stream stream;
// 			vector<function<void( cudaStream_t )>> _;
// 			bool first = true;
// 			friend struct Task;
// 		};
// #endif

	public:
// #ifndef __CUDACC__
// 		using Future = koi::future::_::FutureExt<FutureImpl>;
// #endif

		Task() = default;
		Task( function<void( cudaStream_t )> &&_ ) :
		  _{ std::move( _ ) }
		{
		}

// #ifndef __CUDACC__
// 		Future launch_async( Stream const &stream = Stream() ) &&
// 		{
// 			return Future( FutureImpl( stream, std::move( _ ) ) );
// 		}
// #endif
		Result launch( Stream const &stream = Stream() ) &&
		{
			Event start, stop;
			auto lock = stream.lock();
			start.record();
			for ( auto &e : this->_ ) e( lock.get() );
			stop.record();
			return stop.wait();
		}
		std::future<Result> launch_async( Stream const &stream = Stream() ) &&
		{
			Event start, stop;
			{
				auto lock = stream.lock();
				start.record();
				for ( auto &e : this->_ ) e( lock.get() );
				stop.record();
			}
			return std::async( std::launch::deferred, [=]{ return stop.wait(); } );
		}
		Task &chain( Task &&other )
		{
			for ( auto &e : other._ ) _.emplace_back( std::move( e ) );
			return *this;
		}

	private:
		vector<function<void( cudaStream_t )>> _;
	};

	struct PendingTasks
	{
		PendingTasks &add( future<Result> &&one )
		{
			_.emplace_back( std::move( one ) );
			return *this;
		}
		vector<Result> wait()
		{
			vector<Result> ret;
			for ( auto &e : _ ) {
				e.wait();
				ret.emplace_back( e.get() );
			}
			_.clear();
			return ret;
		}

	private:
		vector<future<Result>> _;
	};
}

VM_END_MODULE()

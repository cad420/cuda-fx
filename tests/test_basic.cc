#include <gtest/gtest.hpp>
#include <cudafx/device.hpp>>

TEST( test_basic, test_basic )
{
	cufx::PendingTasks tasks;

	auto devices = cufx::Devices::scan();
	if ( devices.size() ) {
		auto &device = devices[ 0 ];
		cufx::Image<> image( 1024, 1024 );
		auto view = image.view();
	}
}

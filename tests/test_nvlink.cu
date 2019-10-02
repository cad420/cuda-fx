#include <gtest/gtest.h>

__global__ void f()
{
}

TEST( test_nvlink, test_nvlink )
{
	f<<<1, 1>>>();
}
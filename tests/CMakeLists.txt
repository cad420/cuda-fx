find_package(GTest REQUIRED)

file(GLOB_RECURSE CUDAFX_TEST_SOURCES
  *.cc
  *.cpp
  *.cu
)
set(CUDA_NVCC_FLAGS
  -std=c++14
  #-v
  --expt-extended-lambda
  --use_fast_math
  --keep-device-functions
  -keep
  -maxrregcount
  32
  #-w
  -lineinfo
  -Xcudafe
  --display_error_number
)
cuda_add_executable(cudafx_test_all ${CUDAFX_TEST_SOURCES})
vm_target_dependency(cudafx_test_all cudafx PRIVATE)

if (NOT CMAKE_CXX_COMPILER MATCHES MSVC)
  target_link_libraries(cudafx_test_all pthread)
endif()
target_link_libraries(cudafx_test_all gtest_main gtest)

add_test(cudafx_test_all cudafx_test_all)

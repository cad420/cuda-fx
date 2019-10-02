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

cuda_add_executable(device_inspect 
  device_inspect.cc
)
vm_target_dependency(device_inspect cudafx PRIVATE)

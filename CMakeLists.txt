cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 14)
project(cudafx)

option(CUDAFX_BUILD_TESTS "Set to ON to build tests" OFF)
option(CUDAFX_BUILD_TOOLS "Set to ON to build tools" OFF)

# find cuda compiler and libs
find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

# install build scripts
find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
# install external modules
include(scripts/external.cmake)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/VMUtils
  GIT_TAG        master
)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/stbi
  GIT_TAG        master
)
# export targets
include(export.cmake)

# build tools
if (CUDAFX_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# build tests
if (CUDAFX_BUILD_TESTS)
  add_subdirectory(tests)
endif()

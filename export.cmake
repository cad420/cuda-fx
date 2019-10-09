add_library(cudafx src/dummy.cc)

# target_link_libraries(cudafx ${CUDA_CUDA_LIBRARY})  # if need driver lib
target_include_directories(cudafx INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}	
)
vm_target_dependency(cudafx VMUtils INTERFACE)
vm_target_dependency(cudafx stbi PUBLIC)

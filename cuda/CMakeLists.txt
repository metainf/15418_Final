#CMakeLists.txt to build gpu path tracer
cmake_minimum_required(VERSION 2.8)

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -std=c++11 - O3 -gencode arch=compute_61, code=sm_61
  )

cuda_add_library(
  cuda_lib SHARED
  hello.cu
  test.cu
  )

target_link_libraries(cuda_lib
  ${CUDA_LIBRARIES}
  )

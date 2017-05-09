#CMakeLists.txt to build gpu path tracer
cmake_minimum_required(VERSION 2.8)

project(GPU)

find_package(CUDA QUIET REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 - O3")
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61, code=sm_61)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

file(GLOB GPUSRC
  "test.cu"
  )

CUDA_ADD_LIBRARY(
  GPU SHARED ${GPUSRC}
  )

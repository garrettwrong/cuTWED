# CMakeLists.txt

cmake_minimum_required(VERSION 3.8.0)
project(cuTWED)

set(CFLAGS $(CFLAGS) -Wall -Wextra)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)


include_directories(src)
cuda_add_library(cuTWED SHARED src/cuTWED.cu)
target_link_libraries(cuTWED cublas)

cuda_add_executable(test.x tests/test.cu)
target_link_libraries(test.x cuTWED)

cuda_add_executable(testf.x tests/testf.cu)
target_link_libraries(testf.x cuTWED)

add_library(twed SHARED reference_implementation/twed.c)
add_executable(twed.x reference_implementation/twed_test.c)
target_link_libraries(twed m)
target_link_libraries(twed.x twed)

# CMake doens't appear to grok the cu files as well as C, sigh, help it out
add_custom_command(
  OUTPUT cuTWED.h.i
  COMMAND "${CMAKE_C_COMPILER}" -P -E "${CMAKE_CURRENT_SOURCE_DIR}/src/cuTWED.h" -o "${CMAKE_CURRENT_SOURCE_DIR}/cuTWED/cuTWED.h.i"
  MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/src/cuTWED.h"
  COMMENT "Prepocessing cuTWED.h"
  VERBATIM
  )
add_custom_target(
  PreProcessCFFIHeaders ALL
  DEPENDS cuTWED.h.i
  COMMENT "Processing C Headers for use with Python CFFI"
  VERBATIM
  )

# so cffi can find it with less user intervention
add_custom_command(
  TARGET cuTWED POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_BINARY_DIR}/libcuTWED.so
  ${CMAKE_SOURCE_DIR}/cuTWED/libcuTWED.so
  )


# make install stuff
install(TARGETS cuTWED DESTINATION lib)
install(FILES src/cuTWED.h DESTINATION include)
install(FILES src/cuTWED_core.h DESTINATION include)

cmake_minimum_required(VERSION 3.21)
project(fftx_sample C)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

set(CMAKE_C_STANDARD 99)

find_package(FFTW3F REQUIRED)
include_directories(${FFTW3F_INCLUDE_DIRS})
link_directories(${FFTW3F_LIBRARY_DIRS})

add_executable(fftx_sample main.c)
target_link_libraries(fftx_sample  m ${FFTW3F_LIBRARIES})

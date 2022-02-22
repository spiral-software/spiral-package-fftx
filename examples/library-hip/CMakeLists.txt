##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.14 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )
set ( PROJECT unithipdriver )	## unithipdriver
project ( ${PROJECT}
         VERSION 1.0.1
                DESCRIPTION "Driver for hip mddft transforms of different sizes" )
##                  LANGUAGES C CXX ) CUDA )

##  Add additional paths for CMake to find HIP
list ( APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm )	##  /opt/rocm-4.2.0/hip

find_package ( hip REQUIRED )
if (${hip_FOUND})
    ##  HIP found
    message ( STATUS "Found HIP: Version = ${hip_VERSION}")
else ()
    message ( SEND_ERROR "HIP NOT FOUND: HIP is required to build/run examples")
endif ()

##  Get SPIRAL home...

if ( DEFINED ENV{SPIRAL_HOME} )
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

##  Set compiler and linker.
if ( NOT WIN32 )
    set ( CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE} )
    set ( CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE} )
endif ()

##  Source file 
set ( SOURCES ${PROJECT}.cpp )

##  Need to add $ROCM_PATH for includes and libraries
if ( DEFINED ENV{ROCM_PATH} )
    message ( STATUS "ROCM_PATH is defined: $ENV{ROCM_PATH}" )
    include_directories ( $ENV{ROCM_PATH}/hipfft/include $ENV{ROCM_PATH}/include )
endif ()

##  set ( _local_dir $ENV{HOME}/.local )
##  message ( STATUS "Local directory for rocfft & hipfft includes & libraries: ${_local_dir}" )
include_directories ( ${CMAKE_SOURCE_DIR}
                      ${CMAKE_BINARY_DIR} 
##                        ${_local_dir}/hipfft/include
##                        ${_local_dir}/rocfft/include
                      ${SPIRAL_SOURCE_DIR}/profiler/targets/include )

##  Setup compilation flags
##  Parameters are defined on the cmake command line to specify names & sizes for the driver program
##  INCLUDE_DFT -- name of the file containing the generated code, included by driver program
##  FUNCNAME -- name of the kernel to call
##  DIM_M -- 1st dimension (slowest changing in C world)
##  DIM_N -- 2nd dimension
##  DIM_K -- 3rd dimension
##  USE_DIFF_DATA -- optional define, test its presence in order not to emit hanging -D

if ( "X${USE_DIFF_DATA}" STREQUAL "X" )
    set ( _use_diff )
else ()
    set ( _use_diff "-DUSE_DIFF_DATA" )
endif ()

if (WIN32)
    set ( HIP_COMPILE_FLAGS -include ${INCLUDE_DFT} )
    set ( LIBS_FOR_HIP hipfft rocfft )
    list ( APPEND ADDL_COMPILE_FLAGS -DWIN64 -DFUNCNAME=${FUNCNAME} -DM=${DIM_M} -DN=${DIM_N} -DK=${DIM_K} ${_use_diff} )
else ()
    set ( HIP_COMPILE_FLAGS -include ${INCLUDE_DFT} -g )
    set ( LIBS_FOR_HIP hipfft rocfft )
    list ( APPEND ADDL_COMPILE_FLAGS -DFUNCNAME=${FUNCNAME} -DM=${DIM_M} -DN=${DIM_N} -DK=${DIM_K} ${_use_diff} )
endif ()

message ( STATUS "PROJECT = ${PROJECT}, source file = ${PROJECT}.cpp" )

set                        (   EXE_TARGET  ${PROJECT}-${DIM_M}x${DIM_N}x${DIM_K} )
add_executable		   ( ${EXE_TARGET} ${PROJECT}.cpp )
target_compile_options     ( ${EXE_TARGET} PRIVATE ${HIP_CCOMPILE_FLAGS} ${ADDL_COMPILE_FLAGS} )

target_link_directories ( ${EXE_TARGET} PRIVATE $ENV{ROCM_PATH}/lib )
target_link_libraries	   ( ${EXE_TARGET} hip::host hip::device ${LIBS_FOR_HIP} )

install ( TARGETS
    ${EXE_TARGET}
    DESTINATION ${CMAKE_SOURCE_DIR}/executables )



##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.14 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )
set ( PROJECT dftdriver )
project ( ${PROJECT}
         VERSION 1.0.1
                DESCRIPTION "Driver for mdprdft transforms of different sizes"
                LANGUAGES C CXX CUDA )

set ( SOURCES ${PROJECT}.cu )

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

include_directories ( ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
                      ${SPIRAL_SOURCE_DIR}/profiler/targets/include )

##  Setup compilation flags
##  Parameters are defined on the cmake command line to specify names & sizes for the driver program
##  INCLUDE_DFT -- name of the file containing the generated code, included by driver program
##  FUNCNAME -- name of the kernel to call
##  DIM_M -- 1st dimension (slowest changing in C world)
##  DIM_N -- 2nd dimension
##  DIM_K -- 3rd dimension

if (WIN32)
    set ( CUDA_COMPILE_FLAGS -rdc=false --pre-include ${INCLUDE_DFT} )
    set ( GPU_COMPILE_DEFNS -Xptxas -v -maxrregcount=64 )			## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft )
    list ( APPEND ADDL_COMPILE_FLAGS -DWIN64 -DFUNCNAME=${FUNCNAME} -DM=${DIM_M} -DN=${DIM_N} -DK=${DIM_K} )
else ()
    set ( CUDA_COMPILE_FLAGS -m64 -rdc=true --pre-include ${INCLUDE_DFT} )
    set ( GPU_COMPILE_DEFNS -dc -Xptxas -v -maxrregcount=64 )		## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft_static culibos )
    list ( APPEND ADDL_COMPILE_FLAGS -DFUNCNAME=${FUNCNAME} -DM=${DIM_M} -DN=${DIM_N} -DK=${DIM_K} )
endif ()

##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
set ( CMAKE_CUDA_ARCHITECTURES 70 )

set                        (   EXE_TARGET  ${PROJECT}-${DIM_M}x${DIM_N}x${DIM_K} )  ##  -s-${RSEED} )
add_executable		   ( ${EXE_TARGET} ${PROJECT}.cu )
target_compile_definitions ( ${EXE_TARGET} PRIVATE ${ADDL_COMPILE_FLAGS} )
target_compile_options     ( ${EXE_TARGET} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )
##  if ( NOT WIN32 )
##      target_compile_options ( ${PROJECT} PRIVATE -ccbin=${CMAKE_CXX_COMPILER} )
##  endif ()

set_property        ( TARGET ${EXE_TARGET} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_libraries	   ( ${EXE_TARGET} ${LIBS_FOR_CUDA} )

install ( TARGETS
    ${EXE_TARGET}
    DESTINATION ${CMAKE_SOURCE_DIR}/exes )

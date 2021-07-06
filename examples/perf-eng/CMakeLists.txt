##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.14 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )
set ( PROJECT transformlib_test )    ##  driver to test library of pre-compiled mddft's
project ( ${PROJECT}
         VERSION 1.0.1
                DESCRIPTION "Driver to test library of pre-compiled mddft transforms"
                LANGUAGES C CXX CUDA )

##  Set C standard 11
set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

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

include_directories ( ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/lib_mddft3d_srcs
                      ${SPIRAL_SOURCE_DIR}/profiler/targets/include )

##  Setup compilation flags

if (WIN32)
    ##  set ( CUDA_COMPILE_FLAGS )		##   -rdc=false [this is default]
    set ( GPU_COMPILE_DEFNS -Xptxas -v -maxrregcount=64 )			## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft )
    list ( APPEND ADDL_COMPILE_FLAGS -DWIN64 )
else ()
    set ( CUDA_COMPILE_FLAGS -m64 -rdc=false )		##   -m64 -rdc=false [ 64bit, non-relocatable, default]
    set ( GPU_COMPILE_DEFNS -Xptxas -v -maxrregcount=64 )		## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft culibos )
    list ( APPEND ADDL_COMPILE_FLAGS )
endif ()

##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
set ( CMAKE_CUDA_ARCHITECTURES 70 )

##  Walk the following subdirectory for the source files to add to the library
add_subdirectory ( lib_mddft3d_srcs )		## lib-srcs

set                        ( _lib_precomp mddft3d_precomp )
set                        (   EXE_TARGET  ${PROJECT} )
add_executable		   ( ${EXE_TARGET} ${PROJECT}.cu )
target_compile_definitions ( ${EXE_TARGET} PRIVATE ${ADDL_COMPILE_FLAGS} )
target_compile_options     ( ${EXE_TARGET} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )

set_property        ( TARGET ${EXE_TARGET} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_directories    ( ${EXE_TARGET} PRIVATE ${CMAKE_BINARY_DIR}/bin )
target_link_libraries	   ( ${EXE_TARGET} ${LIBS_FOR_CUDA} ${_lib_precomp} )
add_dependencies           ( ${EXE_TARGET} ${_lib_precomp} )

install ( TARGETS
    ${EXE_TARGET}
    DESTINATION ${CMAKE_BINARY_DIR}/bin )

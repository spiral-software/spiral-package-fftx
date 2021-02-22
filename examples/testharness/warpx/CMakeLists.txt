##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.14 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )

if ( "X${_harnessName}" STREQUAL "X" )
    message ( FATAL_ERROR "Test harnes program name must be defined" )
else ()
    set ( PROJECT ${_harnessName} )
endif ()

project ( ${PROJECT} 
         VERSION 1.0.1
                DESCRIPTION "Test harness for spiral code"
                LANGUAGES C CXX CUDA )

set ( SOURCES ${PROJECT}.cu )

##  Get SPIRAL home...

if ( DEFINED ENV{SPIRAL_HOME} )
    ##  SPIRAL_HOME is defined - use it
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    ##  SPIRAL_HOME is not defined in environment, check if passed on command line
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        ##  not defined on command line -- assume this is a spiral package in
	##  <spiral>/namespaces/packages/<pkg>/examples/testharness/<example>
        message ( STATUS "SPIRAL_HOME undefined, assuming location <spiral>/namespaces/packages/<pkg>/examples/testharness/<example>" )
        set ( SPIRAL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.. )
    else ()
        ##  SPIRAL_HOME defined on command line, use it
        message ( STATUS "SPIRAL_HOME = ${SPIRAL_HOME}" )
        set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
    endif ()
endif ()

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/RunSpiral.cmake")

##  Get the script name(s) and setup the source code generation from scripts

if ( "X${_gpuGapScript}" STREQUAL "X" )
    message ( FATAL_ERROR "GPU Gap script to build/test must be defined" )
endif ()

if ( "X${_gpuFileName}" STREQUAL "X" )
    message ( FATAL_ERROR "Output GPU source code file name for generated code must be defined" )
endif ()

file ( TO_NATIVE_PATH ${CMAKE_SOURCE_DIR}/${_gpuGapScript} _gfile )
create_source_file ( ${_gfile} ${_gpuFileName} )

include_directories ( ${CMAKE_SOURCE_DIR} "Utils" ${SPIRAL_SOURCE_DIR}/profiler/targets ${CMAKE_BINARY_DIR} )

##  Setup compilation flags

if ( NOT "X${_buildCpuCode}" STREQUAL "X" )
    if ( "X${_cpuGapScript}" STREQUAL "X" )
	message ( FATAL_ERROR "Run CPU code specified but CPU Gap script to build/test *not* defined" )
    endif ()

    if ( "X${_cpuFileName}" STREQUAL "X" )
	message ( FATAL_ERROR "Output CPU source code file name for generated code must be defined" )
    endif ()
    
    file ( TO_NATIVE_PATH ${CMAKE_SOURCE_DIR}/${_cpuGapScript} _gfile )
    create_source_file ( ${_gfile} ${_cpuFileName} )
    set ( ADDL_COMPILE_FLAGS -DRUN_CPU_CODE )
endif ()

if (WIN32)
    set ( CUDA_COMPILE_FLAGS -rdc=false )
    set ( GPU_COMPILE_DEFNS -Xptxas -v -maxrregcount=64 )			## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft )
    list ( APPEND ADDL_COMPILE_FLAGS -DWIN64 )
else ()
    set ( CUDA_COMPILE_FLAGS -m64 -rdc=true )
    set ( GPU_COMPILE_DEFNS -dc -Xptxas -v -maxrregcount=64 )		## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft_static culibos )
endif ()

##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
set ( CMAKE_CUDA_ARCHITECTURES 70 )

add_custom_target ( Name.${PROJECT_NAME}.build_srcs ALL
    DEPENDS ${_gpuFileName} ${_cpuFileName}
    VERBATIM )

add_executable		   ( ${PROJECT} ${SOURCES} )   ##  ${_gpuFileName} ${_cpuFileName} 
add_dependencies           ( ${PROJECT} Name.${PROJECT_NAME}.build_srcs )

target_compile_definitions ( ${PROJECT} PRIVATE ${ADDL_COMPILE_FLAGS} )
target_compile_options     ( ${PROJECT} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )
if ( NOT WIN32 )
    target_compile_options ( ${PROJECT} PRIVATE -ccbin=${CMAKE_CXX_COMPILER} )
endif ()

set_property        ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_libraries	   ( ${PROJECT} ${LIBS_FOR_CUDA} )

install ( TARGETS
    ${PROJECT}
    DESTINATION ${CMAKE_SOURCE_DIR} )

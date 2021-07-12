##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

# specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required(VERSION 3.14)

##  Get SPIRAL home...

if ( DEFINED ENV{SPIRAL_HOME} )
    ##  SPIRAL_HOME is defined - use it
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    ##  SPIRAL_HOME is not defined, check if passed on command line
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        ##  not defined on command line -- assume this is a package in <spiral>/namespaces/packages
        message ( STATUS "SPIRAL_HOME undefined, assuming location <spiral>/namespaces/packages/<pkg>" )
        set ( SPIRAL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../.. )
    else ()
        ##  SPIRAL_HOME defined on command line, use it
        message ( STATUS "SPIRAL_HOME = ${SPIRAL_HOME}" )
        set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
    endif ()
endif ()

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/RunTestTarget.cmake")

##  Find python3, i.e., verify exists and get executable name

find_package (Python3 COMPONENTS Interpreter)
if (${Python3_FOUND})
    ##  It exists, executable is ${Python3_EXECUTABLE}
    message ( STATUS "Found Python3: Version = ${Python3_VERSION}, Executable = ${Python3_EXECUTABLE}")
else ()
    message ( SEND_ERROR "Python3 NOT FOUND: Python is required to run tests")
endif ()

##  Python script to run tests is managed (and installed) by spiral-software

##  Subdirectories holding examples

add_subdirectory ( examples/library-c )

string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "ppc64le" PPC_LINUX)
if ( NOT ${PPC_LINUX} )
    ##  ppc64le (summit) - don't run GPU examples from login node
    add_subdirectory ( examples/library-cuda )
endif ()

find_package ( OpenMP )
if ( ${OpenMP_FOUND} )
    ##  OpenMP support found
    string ( COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "AMD64" INTEL_WIN )
    string ( COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "x86_64" INTEL_LINUX )
    if ( ${INTEL_LINUX} OR ${INTEL_WIN} )
	##  Intel architecture, include the AVX/OMP tests
	add_subdirectory ( examples/library-avx-omp )
    endif ()
endif ()

##  Don't add this by default -- generates HIP examples and depends on hip environment
##  add_subdirectory ( examples/library-hip )

enable_testing ()

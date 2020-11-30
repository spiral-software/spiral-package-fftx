##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

# specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required(VERSION 3.14)

# set(CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

# project(SPIRAL_EXAMPLES
#         VERSION 8.2.1
#                 DESCRIPTION "SPIRAL Examples Project"
#                 LANGUAGES C CXX)
                
# ##  Prevent building directly into the source tree
# string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" SPIRAL_COMPILE_INPLACE)
# if (SPIRAL_COMPILE_INPLACE)
#     if (WIN32)
#         message ( FATAL_ERROR, "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
#                  rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
#                  mkdir <build>
#                  cd <build>
#                  cmake ..
#                  then build the solution with VS")
# ##                 cmake --build . --config <config> --target <target> 
#     else()
#         message ( FATAL_ERROR, "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
#                  rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
#                  mkdir <build>
#                  cd <build>
#                  cmake ..
#                  make | make install")
#     endif()
# endif()
  
# if ((NOT DEFINED CMAKE_BUILD_TYPE) OR (NOT CMAKE_BUILD_TYPE))
#     set(CMAKE_BUILD_TYPE Release)
# endif()

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
add_subdirectory ( examples/library-cuda )

enable_testing ()

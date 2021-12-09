##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required (VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  Each example by definition illustrates a concept or technique.  However, the
##  example files are not necessarily complete tests in and of themselves, it
##  may be necessary to pass additional files to SPIRAL to build out a test
##  (e.g., configuration or sizing parameters or specific instructions for a
##  test like CVector).

set ( _hip_examples
    )

set ( _hip_nocheck
      dft-batch-hip.g
      mddft-hip.g
    )

##  _pre_fils are files to preceed that example script; _post_fils are files to
##  follow the script.

set ( _gpu_required_flag True )
set ( _pre_fils "" )
set ( _post_fils "../library-cuda/measures.g" "../library-cuda/vector-add.g" )

foreach ( _example ${_hip_examples} )
    set ( _inp_fils ${_pre_fils} ${_example} ${_post_fils} )
    RunTestTarget( ${_gpu_required_flag} ${_example} ${CMAKE_CURRENT_SOURCE_DIR} ${_inp_fils} )
endforeach ()

##  Currently can't automatically check example expecting an input symbol, do codegen only

set ( _post_fils "" )
foreach ( _example ${_hip_nocheck} )
    set ( _inp_fils ${_pre_fils} ${_example} ${_post_fils} )
    RunTestTarget( ${_gpu_required_flag} ${_example} ${CMAKE_CURRENT_SOURCE_DIR} ${_inp_fils} )
endforeach ()

set_tests_properties (${_hip_examples} PROPERTIES LABELS "HIP examples")
set_tests_properties (${_hip_nocheck}  PROPERTIES LABELS "HIP examples")

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

enable_testing ()


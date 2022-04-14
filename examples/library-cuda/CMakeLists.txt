##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required (VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  GPU/CUDA examples source directory
set ( GPU_EXAMPLES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )

##  Each example by definition illustrates a concept or technique.  However, the
##  example files are not necessarily complete tests in and of themselves, it
##  may be necessary to pass additional files to SPIRAL to build out a test
##  (e.g., configuration or sizing parameters or specific instructions for a
##  test like CVector).

set ( _gpu_examples
      dft-batch-cuda.g
      imdprdft-cuda.g
      mddft-cuda.g
      mddft-batch-cuda.g
      mdprdft-batch-cuda.g
      mdprdft-cuda.g
      mdprdft-pruned-cuda.g
      prdft-batch-cuda.g
    )
  
set ( _gpu_nocheck
      hockney-mlc-cuda.g
      mdrconv-batch-cuda.g
      warpx-cuda.g
    )

##  _pre_fils are files to preceed that example script; _post_fils are files to
##  follow the script.

set ( _pre_fils "" )

set ( _gpu_required_flag True )
set ( _post_fils "measures.g" "vector-add.g" )
message ( STATUS "Loop over the examples: ${_gpu_examples}" )

foreach ( _example ${_gpu_examples} )
    set ( _inp_fils ${_pre_fils} ${_example} ${_post_fils} )
    RunTestTarget( ${_gpu_required_flag} ${_example} ${GPU_EXAMPLES_SOURCE_DIR} ${_inp_fils} )
endforeach ()

##  Currently can't automatically check examples expecting an input symbol, do codegen only

set ( _post_fils "" )
foreach ( _example ${_gpu_nocheck} )
    set ( _inp_fils ${_pre_fils} ${_example} ${_post_fils} )
    RunTestTarget( ${_gpu_required_flag} ${_example} ${GPU_EXAMPLES_SOURCE_DIR} ${_inp_fils} )
endforeach ()

set_tests_properties (${_gpu_examples} PROPERTIES LABELS "GPU examples")
set_tests_properties (${_gpu_nocheck}  PROPERTIES LABELS "GPU examples")

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

enable_testing ()


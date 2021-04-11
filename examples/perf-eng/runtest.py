#! python

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

##  This script reads a file, cube-sizes.txt, that contains several cube size
##  specifications for the 3D DFT.  The script will generate the CUDA code for the DFT and
##  then compile it and run it using the spiral profiler and the DFT harness driver.

import sys
import subprocess
import os
import re

_inclfile = 'mddft3d.cu'
_funcname = 'mddft3d'

with open ( 'cube-sizes.txt', 'r' ) as fil:
    for line in fil.readlines():
        print ( 'Line read = ' + line )
        testscript = open ( 'testscript.g', 'w' )
        testscript.write ( line )
        testscript.close()
        line = re.sub ( '.*\[', '', line )               ## drop "szcube := ["
        line = re.sub ( '\].*', '', line )               ## drop "];"
        line = re.sub ( ' *', '', line )                 ## compress out white space
        line = line.rstrip()                             ## remove training newline
        dims = re.split ( ',', line )
        _dimx = dims[0]
        _dimy = dims[1]
        _dimz = dims[2]

        ##  Generate the SPIRAL script: cat testscript.g, mddft-cuda-frame.g, and mddft-meas.g
        _spiralhome = os.environ.get('SPIRAL_HOME')
        _catfils = _spiralhome + '/gap/bin/catfiles.py'
        cmdstr = 'python ' + _catfils + ' myscript.g testscript.g mddft-cuda-frame.g mddft-meas.g'
        result = subprocess.run ( cmdstr, shell=True, check=True )
        res = result.returncode

        ##  Generate the code by running SPIRAL
        if sys.platform == 'win32':
            cmdstr = _spiralhome + '/spiral.bat < myscript.g'
        else:
            cmdstr = _spiralhome + '/spiral < myscript.g'
            
        result = subprocess.run ( cmdstr, shell=True, check=True )
        res = result.returncode

        ##  Create the build directory (if it doesn't exist)
        build_dir = 'build'
        isdir = os.path.isdir ( build_dir )
        if not isdir:
            os.mkdir ( build_dir )

        ##  Run cmake and build to build the code

        cmdstr = 'rm -rf * && cmake -DINCLUDE_DFT=' + _inclfile + ' -DFUNCNAME=' + _funcname
        cmdstr = cmdstr + ' -DDIM_M=' + _dimx + ' -DDIM_N=' + _dimy + ' -DDIM_K=' + _dimz

        os.chdir ( build_dir )

        if sys.platform == 'win32':
            cmdstr = cmdstr + ' .. && cmake --build . --config Release --target install'
        else:
            cmdstr = cmdstr + ' .. && make install'

        print ( cmdstr )
        ##  sys.exit (0)                    ## testing script, stop here

        result = subprocess.run ( cmdstr, shell=True, check=True )
        res = result.returncode

        if (res != 0):
            print ( result )
            sys.exit ( res )

        os.chdir ( '..' )

        if sys.platform == 'win32':
            cmdstr = './dftdriver.exe'
        else:
            cmdstr = './dftdriver'
    
        result = subprocess.run ( cmdstr )
        res = result.returncode

        if (res != 0):
            print ( result )
            sys.exit ( res )

sys.exit (0)



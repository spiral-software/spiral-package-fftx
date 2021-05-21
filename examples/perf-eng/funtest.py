#! python

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

##  This script reads a file, cube-sizes.txt, that contains several cube size
##  specifications for the 3D DFT.  This script will:
##      Generate a SPIRAL script using the DFT size spec
##      Run SPIRAL to generate the CUDA code
##      Save the generated code (to sub-directory srcs)
##      Compile the CUDA code for the DFT (install in sub-directory exes)
##      Build a script to run all the generated code (timescript.sh)
##  

import sys
import subprocess
import os, stat
import re
import shutil
import time

_inclfile = 'mddft3d.cu'
_funcname = 'mddft3d'
_timescript = 'timescript.sh'

##  Setup 'empty' timing script
timefd = open ( _timescript, 'w' )
timefd.write ( '#! /bin/bash \n\n' )
timefd.write ( '##  Timing script to run Cuda code for various transform sizes \n\n' )
timefd.close()
_mode = stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
os.chmod ( _timescript, _mode )

with open ( 'cube-sizes2.txt', 'r' ) as fil:
    for line in fil.readlines():
        print ( 'Line read = ' + line )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue

        line = re.sub ( '.*\[', '', line )               ## drop "szcube := ["
        line = re.sub ( '\].*', '', line )               ## drop "];"
        line = re.sub ( ' *', '', line )                 ## compress out white space
        line = line.rstrip()                             ## remove training newline
        dims = re.split ( ',', line )
        _dimx = dims[0]
        _dimy = dims[1]
        _dimz = dims[2]

        ##  Create the srcs directory (if it doesn't exist)
        srcs_dir = 'srcs'
        isdir = os.path.isdir ( srcs_dir )
        if not isdir:
            os.mkdir ( srcs_dir )

        ##  Copy the files from srcs to local mddft3d.cu
        _filenamestem = '-' + _dimx + 'x' + _dimy + 'x' + _dimz
        _srcfile = 'srcs/' + _funcname + _filenamestem + '.cu'
        shutil.copy ( _srcfile, _inclfile )

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

        timefd = open ( _timescript, 'a' )
        timefd.write ( '##  Cube = [ ' + _dimx + ', ' + _dimy + ', ' + _dimz + ' ]\n' )
        timefd.write ( 'echo "Run size = [ ' + _dimx + ', ' + _dimy + ', ' + _dimz + ' ]"\n' )
        if sys.platform == 'win32':
            _exename = './exes/dftdriver' + _filenamestem + '.exe'
        else:
            _exename = './exes/dftdriver' + _filenamestem

        timefd.write ( _exename +  '\n\n' )
        timefd.close()
        
        time.sleep(1)
        ##  Uncomment this section if you want to run (time) each DFT as it is generated
        # result = subprocess.run ( _exename )
        # res = result.returncode
        # if (res != 0):
        #     print ( result )
        #     sys.exit ( res )

sys.exit (0)



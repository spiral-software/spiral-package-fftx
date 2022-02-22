#! python

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

##  This script reads a file, cube-sizes2.txt, that contains several cube size
##  specifications for the 3D DFT.  This script will:
##      Use the generated code for the size spec (from srcs folder)
##      Compile the HIP code for the DFT (cmake)
##      Generate a script to run the executables (timescript.sh)
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

_use_diff_data = ' -DUSE_DIFF_DATA=ON'
if len ( sys.argv ) < 2:
    print ( 'Usage: ' + sys.argv[0] + ' [ diff_data ]; defaulting to use same data' )
    _diff_data = ''
else:
    _diff_data = _use_diff_data

##  Setup 'empty' timing script
timefd = open ( _timescript, 'w' )
timefd.write ( '#! /bin/bash \n\n' )
timefd.write ( '##  Timing script to run native HIP for assorted transform sizes \n\n' )
timefd.close()
_mode = stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
os.chmod ( _timescript, _mode )

with open ( 'cube-sizes2.txt', 'r' ) as fil:
    for line in fil.readlines():
        ##  print ( 'Line read = ' + line )
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
        cmdstr = cmdstr + ' -DCMAKE_CXX_COMPILER=hipcc'
        cmdstr = cmdstr + ' -DDIM_M=' + _dimx + ' -DDIM_N=' + _dimy + ' -DDIM_K=' + _dimz + _diff_data

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
        timefd.write ( '##  Size = [' + _dimx + ', ' + _dimy + ', ' + _dimz + ' ]\n' )
        _exename = './executables/hipdriver' + _filenamestem
        if sys.platform == 'win32':
            _exename = _exename + '.exe'

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



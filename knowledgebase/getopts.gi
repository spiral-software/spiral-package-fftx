
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

FFTXGlobals.confGPU := (arg) >> ApplyFunc(arg[1].defaultCUDADeviceConf, Drop(arg,1));
spiral.LocalConfig.fftx := FFTXGlobals;

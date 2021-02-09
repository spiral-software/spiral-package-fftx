FFTXGlobals.confGPU := (arg) >> ApplyFunc(arg[1].defaultCUDADeviceConf, Drop(arg,1));
spiral.LocalConfig.fftx := FFTXGlobals;

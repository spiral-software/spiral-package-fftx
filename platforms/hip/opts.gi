
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

doHIPify := function(opts)
    opts.operations := rec(Print := s -> Print("<FFTX HIPified CUDA options record>"));
    return opts;
end;

Class(FFTXHIPDefaultConf, rec(
    operations := rec(Print := s -> Print("<FFTX FFTX HIPified CUDA Configuration>")),
    useHIP := true
));

hipConf := rec(
    defaultName := "defaultHIPConf",
    defaultOpts := (arg) >> rec(),
    confHandler := doHIPify 
);

fftx.FFTXGlobals.registerConf(hipConf);


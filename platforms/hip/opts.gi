
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

_dim3 := d -> "dim3("::d.id::")";

Class(HIPUnparser, CudaUnparser, rec(
    cu_call := (self, o, i, is) >>
        Print(Blanks(i), "hipLaunchKernelGGL(",  
                self.infix([o.func, _dim3(o.dim_grid), _dim3(o.dim_block), "0", "0"]::o.args, ", "), ");\n"),
));


Class(FFTXHIPOpts, FFTXOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX HIP options record>")),    
    max_threads := 1024
));


doHIPify := function(opts)
    opts.originalCudaOptsID := Copy(opts.operations.Print);
    opts.operations := rec(Print := s -> Print("<FFTX HIPified CUDA options record>"));
    opts.unparser := HIPUnparser;
    opts.includes := ["\"hip/hip_runtime.h\""];
#    opts.postProcessCode := (c, opts) -> FixUpHIP_Code(PingPong_3Stages(c, opts), opts);
    opts.postProcessCode := (c, opts) -> FixUpHIP_Code(c, opts);

    ##  Set options target to "linux-hip" so profiler can build for HIP
    ##  HIP is currently only on Linux but see cuda/opts.gi if a more generic approach to target is ever needed.
    opts.target := rec ( name := "linux-hip" );

    return opts;
end;

Class(FFTXHIPDefaultConf, rec(
    __call__ := self >> self,
    getOpts := (self, t) >> doHIPify(ParseOptsCUDA(FFTXCUDADeviceDefaultConf, t)),
    operations := rec(Print := s -> Print("<FFTX FFTX HIPified CUDA Configuration>")),
    useHIP := true
));

hipConf := rec(
    defaultName := "defaultHIPConf",
    defaultOpts := (arg) >> FFTXHIPDefaultConf,
    confHandler := doHIPify 
);

fftx.FFTXGlobals.registerConf(hipConf);


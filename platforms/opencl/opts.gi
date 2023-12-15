
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

_dim3 := d -> "dim3("::d.id::")";

Class(OpenCLUnparser, CudaUnparser, rec(
    cu_call := (self, o, i, is) >>
        Print(Blanks(i), "hipLaunchKernelGGL(",  
                self.infix([o.func, _dim3(o.dim_grid), _dim3(o.dim_block), "0", "0"]::o.args, ", "), ");\n"),
    
    simtThreadIdxX := (self, o, i, is) >> Print("get_local_id(0)"),
    simtThreadIdxY := (self, o, i, is) >> Print("get_local_id(1)"),
    simtThreadIdxZ := (self, o, i, is) >> Print("get_local_id(1)"),

    simtBlockIdxX := (self, o, i, is) >> Print("get_group_id(0)"),
    simtBlockIdxY := (self, o, i, is) >> Print("get_group_id(1)"),
    simtBlockIdxZ := (self, o, i, is) >> Print("get_group_id(2)"),

    simt_syncblock := (self, o, i, is) >> Print(Blanks(i), "barrier(CLK_LOCAL_MEM_FENCE);\n"),
    simt_synccluster := (self, o, i, is) >> Print(Blanks(i), "barrier(CLK_LOCAL_MEM_FENCE);\n"),
));


Class(FFTXOpenCLOpts, FFTXOpts, rec(
    tags := [],
    operations := rec(Print := s -> Print("<FFTX OpenCL options record>")),    
    max_threads := 1024
));


doOpenCLify := function(opts)
    opts.originalCudaOptsID := Copy(opts.operations.Print);
    opts.operations := rec(Print := s -> Print("<FFTX OpenCLified CUDA options record>"));
    opts.unparser := OpenCLUnparser;
    opts.codegen := OpenCLCodegen;
    # opts.XType := TPtr(opts.XType.t, ["global"]);
    # opts.YType := TPtr(opts.YType.t, ["global"]);
    opts.fixUpTeslaV_Code := true;
    # opts.YPtr.t := TPtr(opts.YPtr.t.t, ["global"]);
    # opts.includes := ["\"hip/hip_runtime.h\""];
#    opts.postProcessCode := (c, opts) -> FixUpHIP_Code(PingPong_3Stages(c, opts), opts);
    opts.postProcessCode := (c, opts) -> FixUpOpenCL_Code(c, opts);

    return opts;
end;

Class(FFTXOpenCLDefaultConf, rec(
    __call__ := self >> self,
    getOpts := (self, t) >> doOpenCLify(ParseOptsCUDA(FFTXCUDADeviceDefaultConf, t)),
    operations := rec(Print := s -> Print("<FFTX OpenCLified CUDA Configuration>")),
    useHIP := true
));

OpenCLConf := rec(
    defaultName := "defaultOpenCLConf",
    defaultOpts := (arg) >> FFTXOpenCLDefaultConf,
    confHandler := doOpenCLify 
);

fftx.FFTXGlobals.registerConf(OpenCLConf);


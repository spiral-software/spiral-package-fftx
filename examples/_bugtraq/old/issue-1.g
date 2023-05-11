##  This script shows a case where codegen is requesting far too much shared memory on the device:
##  Run the script, look in generated code file: fftx_mdprdft_100x100x100.cu
##  Am seeing statements like:      __shared__ double T53[2512500];
##  which is far greater than GPU can handle -- code won't compile.
##  When we fix this we can delve into the issue of correctness that Peter has noted.

szcube := [ 100, 100, 100 ];
szcube := [ 64, 64, 64 ];
libdir := "lib_fftx_mdprdft_srcs"; 
file_suffix := ".cu"; 
fwd := true; 
codefor := "CUDA"; 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA/HIP code to a file.  The code will be compiled into a library for applications
##  to link against -- providing pre-compiled FFTs of standard sizes.

Load(fftx);
ImportAll(fftx);

if codefor = "CUDA" then
    conf := LocalConfig.fftx.confGPU();
else
    conf := FFTXGlobals.defaultHIPConf();
fi;

if fwd then
    prefix := "fftx_mdprdft_";
    prdft  := MDPRDFT;
    sign   := -1;
else
    prefix := "fftx_imdprdft_";
    prdft  := IMDPRDFT;
    sign   := 1;
fi;

if 1 = 1 then
    name := prefix::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("fftx_mdprdft-frame: name = ", name, ", cube = ", szcube, ", size = ",
              StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1),
                                                                    s->" x "::StringInt(s))),
              ";\t\t##PICKME##");

    ## This line from mddft-frame-cuda.g :
    ##    t := TFCall(TRC(MDDFT(szcube, 1)), 
    ##                rec(fname := name, params := []));
    ##  szrevcube := Reversed(szcube);
    szhalfcube := [Int(szcube[1]/2)+1]::Drop(szcube,1);
    ##  szhalfcube := [szcube[1]/2+1]::Drop(szcube,1);
    var_1:= var("var_1", BoxND([0,0,0], TReal));
    var_2:= var("var_2", BoxND(szcube, TReal));
    var_3:= var("var_3", BoxND(szhalfcube, TReal));
    var_2:= X;
    var_3:= Y;
    symvar := var("sym", TPtr(TReal));
    t := TFCall(TDecl(TDAG([
           TDAGNode(TTensorI(prdft(szcube,sign),1,APar, APar), var_3,var_2),
                  ]),
            [var_1]
            ),
        rec(fname:=name, params:= [symvar])
    );
    
    opts := conf.getOpts(t);
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;
    tt := opts.tagIt(t);
    if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;
    opts.printRuleTree := true;
    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo(name::file_suffix, opts.prettyPrint(c));
fi;


##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA code to a file.  The code will be compiled along with a test harness to run the
##  code, timing it against a cufft specification of the same size, and validating that
##  the results are the same for both.

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

if 1 = 1 then
    d := Length(szcube);
    
    name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("mddft-cuda-frame: name = ", name, ", cube = ", szcube, ", size = ",
              StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1),
                                                                    s->" x "::StringInt(s))),
              ";\t\t##PICKME##");
    
    t := TFCall(TRC(MDDFT(szcube, 1)), 
                rec(fname := name, params := []));
    
    opts := conf.getOpts(t);
    ##  PrintLine("DEBUG: opts = ", opts);
    ##  opts.printRuleTree := true;
    if IsBound ( seedme ) then 
        RandomSeed ( seedme );
    fi;
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need to functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;
    tt := opts.tagIt(t);
    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo(libdir::"/"::name::".cu", opts.prettyPrint(c));
fi;

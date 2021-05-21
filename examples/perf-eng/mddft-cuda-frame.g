
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

_stressTest := true;
#_stressTest := false;
_sample := 10;
_cubic := true;

if _stressTest then
    MAX_KERNEL := 16;
    MAX_PRIME := 7;
#    MIN_SIZE := 32;
    MIN_SIZE := 256;
#    MAX_SIZE := 256;
    MAX_SIZE := 320;
#    MAX_SIZE := 1024;

    _thold := MAX_KERNEL;
    filter := (e) -> When(e[1] * e[2] <= _thold ^ 2, e[1] <= _thold and e[2] <= _thold, e[1] <= _thold and e[2] >= _thold);
    size1 := Filtered([MIN_SIZE..MAX_SIZE], i -> ForAny(DivisorPairs(i), filter) and ForAll(Factors(i), j -> not IsPrime(j) or j <= MAX_PRIME));
   
    sizes3 := When(_cubic, List(size1, k -> Replicate(3, k)), Cartesian(Replicate(3, size1)));
    sizes := When(_sample = 0, sizes3, List([1.._sample], i->Random(sizes3)));
else
    sizes := [
     [128, 128, 128] 
#     [ 96, 96, 320],
#     [ 100, 100, 100],
#     [ 224, 224, 100],
#     [ 80, 80, 80 ]
    ];
fi;

##  sizes := [[270, 270, 270]];

##  for szcube in sizes do
if 1 = 1 then
    var.flush();
    d := Length(szcube);
    
    ##  name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

    name := "mddft"::StringInt(d)::"d";
    PrintLine("mddft-cuda-frame: name = ", name, ", cube = ", szcube, ", size = ",
              StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1),
                                                                    s->" x "::StringInt(s))),
              ";\t\t##PICKME##");
    
    ##  PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");
    
    t := TFCall(TRC(MDDFT(szcube, 1)), 
                rec(fname := name, params := []));
    
    opts := conf.getOpts(t);
    PrintLine("DEBUG: opts = ", opts);

    opts.printRuleTree := true;
    if IsBound ( seedme ) then 
        RandomSeed ( seedme );
    fi;

    tt := opts.tagIt(t);
    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo(name::".cu", opts.prettyPrint(c));
fi;
##  od;

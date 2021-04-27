
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

_stressTest := true;
#_stressTest := false;
#_sample := 10;
_sample := 0;
_cubic := true;
#_cubic := false;

if _stressTest then
    MAX_KERNEL := 16;
    MAX_PRIME := 7;
    MIN_SIZE := 32;
#    MIN_SIZE := 256;
#    MAX_SIZE := 256;
    MAX_SIZE := 224;
#    MAX_SIZE := 320;
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

#sizes := [[270, 270, 270]];
sizes := [[32,32,32]];

for szcube in sizes do
    var.flush();
    d := Length(szcube);
    
    name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");
    
    t := TFCall(TRC(MDDFT(szcube, 1)), 
        rec(fname := name, params := []));
    
    opts := conf.getOpts(t);
    PrintLine("DEBUG: opts = ", opts);

    tt := opts.tagIt(t);
    c := opts.fftxGen(tt);
    opts.prettyPrint(c);
    PrintTo(name::".cu", opts.prettyPrint(c));
od;

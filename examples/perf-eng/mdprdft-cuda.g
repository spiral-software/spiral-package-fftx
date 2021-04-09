
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

_stressTest := true;
_sample := 1;

if _stressTest then
    MAX_KERNEL := 16;
    MAX_PRIME := 7;
    MIN_SIZE := 32;
    MAX_SIZE := 320;
    size1 := Filtered([MIN_SIZE..MAX_SIZE], n -> let(fcts := Factors(n), Maximum(fcts) <= MAX_KERNEL and ForAll(fcts, i -> IsPrime(i) and i <= MAX_PRIME)));
   
    sizes3 := Cartesian(Replicate(3, size1));
    sizes := When(_sample = 0, sizes3, List([1.._sample], i->Random(sizes3)));
else
    sizes := [
         [ 100, 100, 100],
         [ 224, 224, 100],
         [ 96, 96, 320],
         [ 80, 80, 80 ]
    ];
fi;

#fwd := true;
fwd := false;

if fwd then
    prdft := MDPRDFT;
    k := 1;
else
    prdft := IMDPRDFT;
    k := -1;
fi;


for szcube in sizes do
    var.flush();
    d := Length(szcube);
    
    name := "mdprdft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("mdprdft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

    t := TFCall(ApplyFunc(prdft, [szcube, k]), 
            rec(fname := name, params := []));
    
    opts := conf.getOpts(t);
    tt := opts.tagIt(t);
    
    c := opts.fftxGen(tt);
    opts.prettyPrint(c);
    PrintTo(name::".cu", opts.prettyPrint(c));
od;

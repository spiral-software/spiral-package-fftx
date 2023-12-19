
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);
Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

fwd := true;
#fwd := false;

N := 16;
#N := 32;
#N := 35;
#N := 272; #has prime factor 17
#N := 128*3;
#N := 80;
#N := 81;
#N := 512;
#N := 128*5;
#N := 64*5;
#N := 16*5*5;
#N := 9*5*5;
#N := 7*9*5;
#N := 768;
#N := 1024;
#N := 640;
N := 648;
szcube :=       Replicate(3, N);

#szcube := [80, 80, 374];
#szcube := [96, 96, 374];
#szcube := [24, 32, 40];
#szcube := [80, 80, 680];

if fwd then
    prdft := MDPRDFT;
    k := 1;
else
    prdft := IMDPRDFT;
    k := -1;
fi;


d := Length(szcube);

name := When(fwd, "", "i")::"mdprdft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("mdprdft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCall(ApplyFunc(prdft, [szcube, k]), 
        rec(fname := name, params := []));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));

CMeasure(c, opts);


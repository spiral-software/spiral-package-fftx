
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

fwd := true;
#fwd := false;

##  szcube :=       [272, 272, 272];
szcube :=       [64, 80, 96];

if fwd then
    prdft := MDPRDFT;
    k := 1;
    xtype := TArrayNDF(TReal, szcube);
    ytype := TArrayNDF_ConjEven(TComplex, szcube);
else
    prdft := IMDPRDFT;
    k := -1;
    xtype := TArrayNDF_ConjEven(TComplex, szcube);
    ytype := TArrayNDF(TReal, szcube);
fi;


d := Length(szcube);

name := When(fwd, "", "i")::"mdprdftf"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

PrintLine("mdprdft-cuda: d = ", d, " cube = ", szcube, ";\t\t##PICKME##");

t := TFCallF(ApplyFunc(prdft, [szcube, k]), 
        rec(fname := name, params := [],
            XType := xtype, YType := ytype));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

_tt := opts.preProcess(tt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo(name::".cu", opts.prettyPrint(c));
CMeasure(c, opts);


##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

d := 3;
szcube := [64, 80, 96];
    
name := "mdrconv"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

symvar := var("sym", TPtr(TReal));

t := TFCallF(IMDPRDFT(szcube, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(szcube, 1))* (Last(szcube)/2+1), 0)) * MDPRDFT(szcube, -1), 
        rec(fname := name, params := [symvar],
        Xtype := TArrayNDF(TReal, szcube), Ytype := TArrayNDF(TReal, szcube)));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);


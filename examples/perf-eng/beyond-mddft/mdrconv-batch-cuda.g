
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);
Import(simt);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

nbatch := 2;
d := 3;
szcube := 64;
ns :=  Replicate(d, szcube);

t := let(batch := nbatch,
    apat := APar,
    name := "rconv"::StringInt(Length(ns))::"d", symvar := var("sym", TPtr(TReal)), 
    TFCall(TTensorI(IMDPRDFT(ns, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(ns, 1))* (Last(ns)/2+1), 0)) * MDPRDFT(ns, -1), batch, apat, apat), 
        rec(fname := name, params := [symvar]))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);


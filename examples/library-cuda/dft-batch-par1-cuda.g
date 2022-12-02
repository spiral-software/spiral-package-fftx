
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

n := 16;
N := 64;

tp := 3; # 1, 2 ,3

tags := [[APar, APar], [APar, AVec], [AVec, APar]];

t := let(
    name := "batchDFT"::StringInt(N)::"type"::StringInt(tp),
    TFCall(TRC(TTensorI(TTensorI(DFT(N, -1), n, APar, APar), n, tags[tp][1], tags[tp][2])), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

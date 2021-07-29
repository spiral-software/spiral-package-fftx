
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 3d batch of 1d real DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := FFTXGlobals.defaultHIPConf();

n := 2;
d := 2;
N := 128;

iter := List([1..d], i->Ind(n));

pdft := When(false,
    PRDFT,
    IPRDFT
);

t := let(
    name :="grid_"::pdft.name::StringInt(d)::"d_cont",
    TFCall(TMap(pdft(N, -1), iter, APar, APar), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);




##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 3d batch of 1d real DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := LocalConfig.fftx.confGPU();

n := 2;
N := 374;

pdft := When(true,
    PRDFT,
    IPRDFT
);

t := let(
    name :="grid_"::pdft.name::StringInt(n),
    TFCall(TTensorI(TTensorI(pdft(N, -1), n, APar, APar), n, APar, APar), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
opts.tags := opts.tags{[1,3]};
opts.breakdownRules.PRDFT[4].maxSize := 17;
tt := opts.tagIt(t);

_tt := opts.preProcess(tt);
rt := opts.search(_tt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

cyc := CMeasure(c, opts);

tm := MatSPL(_tt);
cm := CMatrix(c, opts);

delta := InfinityNormMat(cm-tm);




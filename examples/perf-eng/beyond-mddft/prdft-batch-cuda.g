
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 3d batch of 1d real DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := LocalConfig.fftx.confGPU();

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

# -- testing on Thom ----------------
opts.target.forward := "thom";
opts.target.name := "linux-cuda";

# measurement
cyc := CMeasure(c, opts);
gflops := _gflops(N, n^2, cyc)/2;

# smaller test case -- CMatrix
tm := MatSPL(t);
cm := CMatrix(c, opts);
InfinityNormMat(cm - tm);

# check first column
v := BasisVec(t.dims()[2], 2);

cv := CVector(c, v, opts);
tv := MatSPL(t) * v;
Maximum(cv-tv);








##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

n := 2;
d := 2;
N := 128;

iter := List([1..d], i->Ind(n));

t := let(
    name := "grid_dft"::StringInt(d)::"d_cont",
    TFCall(TRC(TMap(DFT(N, -1), iter, APar, APar)), 
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
gflops := _gflops(N, n^2, cyc);

# smaller test case -- CMatrix
tm := MatSPL(t);
cm := CMatrix(c, opts);
InfinityNormMat(cm - tm);

# check first column
v := BasisVec(t.dims()[2], 2);

cv := CVector(c, v, opts);
tv := MatSPL(t) * v;
Maximum(cv-tv);




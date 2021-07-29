
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);
Import(simt);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

fwd := true;
#fwd := false;

nbatch := 2;
d := 3;
szcube := 64;
ns :=  Replicate(d, szcube);

if fwd then
    prdft := MDPRDFT;
    k := 1;
else
    prdft := IMDPRDFT;
    k := -1;
fi;


t := let(batch := nbatch,
    apat := APar,
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_batch",  
    TFCall(TTensorI(prdft(ns, k), nbatch, apat, apat), 
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
gflops := _gflops(Product(ns), nbatch, cyc)/2;

# check first column
v := BasisVec(t.dims()[2], 0);

cv := CVector(c, v, opts);
if fwd then
    tv := Flat(Replicate(t.dims()[1]/2, [1,0]));
else
    tv := Flat(Replicate(t.dims()[1], 1));
fi;
Maximum(cv-tv);




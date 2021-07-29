
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
szcube := 128;
d := 3;
ns := Replicate(d, szcube);

PrintLine("mddft-batch-cuda: batch = ", nbatch, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, APar, AVec),
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_batch",  
    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
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
gflops := _gflops(Product(ns), nbatch, cyc);

# check first column
v := BasisVec(t.dims()[2], 0);

cv := CVector(c, v, opts);
tv := Flat(Replicate(t.dims()[1]/2, [1,0]));
Maximum(cv-tv);





##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.defaultConf();

nbatch := 512;
szns := [256];

PrintLine("mddft-batch: batch = ", nbatch, " ns = ", szns, ";\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, APar, AVec),
    ns := szns,
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_batch",  
    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);



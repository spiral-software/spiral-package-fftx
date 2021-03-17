
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(packages.powerisa.power9);
Import(packages.powerisa.power9);
Import(packages.powerisa.power9.p9macro);

Load(fftx);
ImportAll(fftx);
LoadImport(fftx.platforms.power);

conf := FFTXGlobals.defaultPOWER9OMPConf();

nbatch := 4;
szns := [16, 16, 16];
name := "dft"::StringInt(Length(szns))::"d_batch_omp";

PrintLine("mddft-batch: batch = ", nbatch, " ns = ", szns, ";\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, APar, AVec),
    ns := szns,
    k := -1,
    name := name,  
    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo(name::".c", opts.prettyPrint(c));



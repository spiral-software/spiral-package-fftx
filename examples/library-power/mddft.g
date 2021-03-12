
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(packages.powerisa.power9);
Import(packages.powerisa.power9);
Import(packages.powerisa.power9.p9macro);

Load(fftx);
ImportAll(fftx);
LoadImport(fftx.platforms.power);

conf := FFTXGlobals.defaultPOWER9Conf();

szns := [16, 16, 16];
name := "dft"::StringInt(Length(szns))::"d";

PrintLine("mddft: ns = ", szns, ";\t\t##PICKME##");

t := let(ns := szns,
    k := -1,
    name := name,  
    TFCall(TRC(MDDFT(ns, k)), rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo(name::".c", opts.prettyPrint(c));


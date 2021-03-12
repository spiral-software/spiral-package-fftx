
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.defaultConf();

szns := [5, 4, 8];

PrintLine("mdprdft: ns = ", szns, ";\t\t##PICKME##");

t := let(ns := szns,
    k := -1,
    dft := When(true, MDPRDFT, IMDPRDFT),
    name := dft.name::StringInt(Length(ns))::"d",  
    TFCall(dft(ns, k), rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);



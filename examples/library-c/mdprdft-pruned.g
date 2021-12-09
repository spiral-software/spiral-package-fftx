
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# Md real pruned DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := LocalConfig.fftx.defaultConf();

szns := [6, 4, 8];
sznzs := 2;

t := let(ns := szns, 
    nzs := sznzs,
    fwd := true,
    ppat := List(ns, n->[1..Int(n/nzs)]),
    blk := 1,
    k := -1,
    name := When(fwd, "", "i")::MDPRDFT.name,  
    TFCall(When(fwd,
        Compose(MDPRDFT(ns, k), ZeroEmbedBox(ns, ppat)),
        Compose(ExtractBox(ns, ppat), IMDPRDFT(ns, k))
    ), rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);


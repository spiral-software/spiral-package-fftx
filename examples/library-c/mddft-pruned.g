# 1d and multidimensional complex pruned DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

t := let(ns := [6, 4, 8],
    nzs := 2,
    fwd := true,
    ppat := List(ns, n->[1..Int(n/nzs)]),
    k := -1,
    name := When(fwd, "", "i")::MDDFT.name::StringInt(Length(ns))::"d",  
    TFCall(TRC(When(fwd,
        Compose(MDDFT(ns, k), ZeroEmbedBox(ns, ppat)),
        Compose(ExtractBox(ns, ppat), MDDFT(ns, k))
    )), rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



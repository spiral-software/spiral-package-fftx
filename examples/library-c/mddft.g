# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

t := let(ns := [5, 4, 8],
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d",  
    TFCall(TRC(MDDFT(ns, k)), rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

t := let(batch := 16,
    apat := When(true, APar, AVec),
    dft := When(true, MDPRDFT, IMDPRDFT),
    ns := [5, 4, 8],
    k := -1,
    name := dft.name::StringInt(Length(ns))::"d_batch",  
    TFCall(TTensorI(dft(ns, k), batch, apat, apat), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



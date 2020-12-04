# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

szns := [5, 4, 8];

PrintLine("mdprdft: ns = ", szns, ";\t\t##PICKME##");

t := let(ns := szns,
    k := -1,
    dft := When(true, MDPRDFT, IMDPRDFT),
    name := dft.name::StringInt(Length(ns))::"d",  
    TFCall(dft(ns, k), rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

nbatch := 4;
szns := [4, 5, 8];

PrintLine("mddft-batch: batch = ", nbatch, " ns = ", szns, ";\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, APar, AVec),
    ns := szns,
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_batch",  
    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



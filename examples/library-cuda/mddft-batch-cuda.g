# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

t := let(batch := 4,
    apat := When(true, APar, AVec),
    ns := [4, 4, 4],
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_batch",  
    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

rt := opts.search(t);
c := opts.fftxGen(t);
opts.prettyPrint(c);



# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

nbatch := 4;
szcube := 4;

PrintLine("mddft-batch-cuda: batch = ", nbatch, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, APar, AVec),
    ns := [szcube, szcube, szcube],
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_batch",  
    TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

rt := opts.search(t);
c := opts.fftxGen(t);
opts.prettyPrint(c);



# 1D batch of 3D PRDFT

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

t := let(batch := 2,
    apat := When(true, AVec, APar),
    dft := When(true, MDPRDFT, IMDPRDFT),
    ns := [80, 80, 80],
    k := -1,
    name := dft.name::StringInt(Length(ns))::"d_batch",  
    TFCall(TTensorI(dft(ns, k), batch, apat, apat), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);




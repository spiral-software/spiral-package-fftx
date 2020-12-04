# 1D batch of 3D PRDFT

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

nbatch := 2;
szcube := 80;

PrintLine("mdprdft-batch-cuda: batch = ", nbatch, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, AVec, APar),
    dft := When(true, MDPRDFT, IMDPRDFT),
    ns := [szcube, szcube, szcube],
    k := -1,
    name := dft.name::StringInt(Length(ns))::"d_batch",  
    TFCall(TTensorI(dft(ns, k), batch, apat, apat), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);




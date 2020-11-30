# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confWarpXCUDADevice();
opts := FFTXGlobals.getOpts(conf);

t := let(batch := 4,
    apat := AVec,
    ns := [80, 80, 80],
    name := "mdprdft_batch"::StringInt(Length(ns))::"d", 
    TFCall(TTensorI(MDPRDFT(ns, 1), batch, apat, apat), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
##  PrintTo("imdprdft80b.c", opts.prettyPrint(c));


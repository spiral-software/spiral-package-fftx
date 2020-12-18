# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confWarpXCUDADevice();
opts := FFTXGlobals.getOpts(conf);

szcube := 80;

PrintLine("mddft-cuda: batch = ", nbatch, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(batch := nbatch,
    apat := AVec,
    ns := [szcube, szcube, szcube],
    name := "mdprdft_batch"::StringInt(Length(ns))::"d", 
    TFCall(TTensorI(MDPRDFT(ns, 1), batch, apat, apat), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
##  PrintTo("imdprdft80b.c", opts.prettyPrint(c));


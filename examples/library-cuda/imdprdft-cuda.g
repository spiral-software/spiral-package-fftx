# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := FFTXGlobals.confWarpXCUDADevice();
opts := FFTXGlobals.getOpts(conf);
opts.printRuleTree := true;

t := let(batch := 2,
    apat := APar,
    n := 80,
    nz := 2,
    ny := 2,
    name := "test", 
    TFCall(TTensorI(TTensorI(TTensorI(IPRDFT(n, 1), ny, apat, apat), nz, apat, apat), batch, apat, apat),
        rec(fname := name, params := [])).withTags(opts.tags)
);

tt := opts.preProcess(t);
rt := opts.search(t);
s := opts.sumsRuleTree(rt);

c := opts.fftxGen(t);
opts.prettyPrint(c);
##  PrintTo("test.c", opts.prettyPrint(c));


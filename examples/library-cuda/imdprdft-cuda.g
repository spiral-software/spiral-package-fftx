# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := FFTXGlobals.confWarpXCUDADevice();
opts := FFTXGlobals.getOpts(conf);
opts.printRuleTree := true;

nbatch := 2;
szn      := 80;
sznz     := 2;
szny     := 2;

PrintLine("imdprdft-batch-cuda: batch = ", nbatch, " n = ", szn, " nz = ", sznz, " ny = ", szny, ";\t\t##PICKME##");

t := let(batch := nbatch,
    apat := APar,
    n := szn,
    nz := sznz,
    ny := szny,
    name := "test", 
    TFCall(TTensorI(TTensorI(TTensorI(IPRDFT(n, 1), ny, apat, apat), nz, apat, apat), batch, apat, apat),
        rec(fname := name, params := [])).withTags(opts.tags)
);

tt := opts.preProcess(t);
rt := opts.search(t);
s := opts.sumsRuleTree(rt);
c := opts.codeSUms(s);


c := opts.fftxGen(t);
opts.prettyPrint(c);
##  PrintTo("test.c", opts.prettyPrint(c));


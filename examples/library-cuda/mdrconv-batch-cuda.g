# Batch Real nD convolution

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confWarpXCUDADevice();
opts := FFTXGlobals.getOpts(conf);

nbatch := 16;
szcube := 80;

PrintLine("mdrconv-batch-cuda: batch = ", nbatch, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(batch := nbatch,
    apat := AVec,
    ns := [szcube, szcube, szcube],
    name := "rconv"::StringInt(Length(ns))::"d", symvar := var("sym", TPtr(TReal)), 
    TFCall(TTensorI(IMDPRDFT(ns, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(ns, 1))* (Last(ns)/2+1), 0)) * MDPRDFT(ns, -1), batch, apat, apat), 
        rec(fname := name, params := [symvar])).withTags(opts.tags)
);

tt := opts.preProcess(t);
rt := opts.search(tt);
s := opts.sumsRuleTree(rt);
##  PrintTo("s-bmdrconv-cuda.g", s);

c := opts.codeSums(s);
##  PrintTo("bmdrconv80.cu", opts.prettyPrint(c));

PrintLine("mdrconv-batch-cuda: codegen test only (no compiled test with 'symbol')\t\t##PICKME##");

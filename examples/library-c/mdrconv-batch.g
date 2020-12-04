# Batch Real nD convolution

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.mdRConv();
opts := FFTXGlobals.getOpts(conf);

cubeside := 80;
nbatch := 4;

PrintLine("mdrconv-batch: batch = ", nbatch, " cube = ", cubeside, "^3;\t\t##PICKME##");

t := let(batch := nbatch,
    apat := When(true, APar, AVec),
    ns := [cubeside, cubeside, cubeside],
    name := "rconv"::StringInt(Length(ns))::"d", symvar := var("sym", TPtr(TReal)), 
    TFCall(TTensorI(IMDPRDFT(ns, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(ns, 1))* (Last(ns)/2+1), 0)) * MDPRDFT(ns, -1), batch, apat, apat), 
        rec(fname := name, params := [symvar])).withTags(opts.tags)
);

tt := opts.preProcess(t);
rt := opts.search(tt);

c := opts.fftxGen(t);
opts.prettyPrint(c);
fname := "bmdrconv"::StringInt(cubeside)::".c";
## PrintTo(fname, opts.prettyPrint(c));

PrintLine("mdrconv-batch: codegen test only (no compiled test with 'symbol')\t\t##PICKME##");

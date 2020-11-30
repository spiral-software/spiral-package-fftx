# Real nD convolution

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.mdRConv();
opts := FFTXGlobals.getOpts(conf);

t := let(ns := [6, 4, 8],
    name := "rconv"::StringInt(Length(ns))::"d", symvar := var("sym", TPtr(TReal)), 
    TFCall(IMDPRDFT(ns, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(ns, 1))* (Last(ns)/2+1), 0)) * MDPRDFT(ns, -1), 
        rec(fname := name, params := [symvar])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);


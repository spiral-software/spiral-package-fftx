# Real nD convolution

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.defaultConf();

szns := [6, 4, 8];

PrintLine("mdrconv: ns = ", szns, ";\t\t##PICKME##");

t := let(ns := szns,
    name := "rconv"::StringInt(Length(ns))::"d", symvar := var("sym", TPtr(TReal)), 
    TFCall(IMDPRDFT(ns, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(ns, 1))* (Last(ns)/2+1), 0)) * MDPRDFT(ns, -1), 
        rec(fname := name, params := [symvar]))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintLine("mdrconv: codegen test only (no compiled test with 'symbol')\t\t##PICKME##");

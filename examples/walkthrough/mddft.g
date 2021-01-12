# 1d and multidimensional complex DFTs

LogTo("walkthrough-mddft-c.txt");

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

szns := [5, 4, 8];

PrintLine("mddft: ns = ", szns, ";\t\t##PICKME##");

t := let(ns := szns,
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d",  
    TFCall(TRC(MDDFT(ns, k)), rec(fname := name, params := [])).withTags(opts.tags)
);

# c := opts.fftxGen(t);
tt := opts.preProcess(t);
rt := opts.search(tt);
s := opts.sumsRuleTree(rt);
c := opts.codeSums(s);

opts.prettyPrint(c);

LogTo();


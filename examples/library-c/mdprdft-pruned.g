# 1d real pruned DFTs
# we do not have the MD nonterminal implemented yet

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

t := let(n := 8, 
    nzs := 2,
    fwd := false,
    ppat := [1..Int(n/nzs)],
    blk := 1,
    k := -1,
    name := When(fwd, "", "i")::PRDFT.name,  
    TFCall(When(fwd,
        Compose(PRDFT(n, k), ZeroEmbedBox(n, ppat)),
        Compose(ExtractBox(n, ppat), IPRDFT(n, k))
    ), rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



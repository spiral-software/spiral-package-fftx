# 1d real pruned DFTs
# we do not have the MD nonterminal implemented yet

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := LocalConfig.fftx.defaultConf();

szn   := 8;
sznzs := 2;

PrintLine("mdprdft-pruned: n = ", szn, " nzs = ", sznzs, ";\t\t##PICKME##");

t := let(n := szn, 
    nzs := sznzs,
    fwd := false,
    ppat := [1..Int(n/nzs)],
    blk := 1,
    k := -1,
    name := When(fwd, "", "i")::PRDFT.name,  
    TFCall(When(fwd,
        Compose(PRDFT(n, k), ZeroEmbedBox(n, ppat)),
        Compose(ExtractBox(n, ppat), IPRDFT(n, k))
    ), rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);



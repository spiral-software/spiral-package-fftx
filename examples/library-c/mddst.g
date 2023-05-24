Load(fftx);
ImportAll(fftx);
Import(dct_dst);

# conf := LocalConfig.fftx.confGPU();
conf := LocalConfig.fftx.defaultConf();

n := 15;
t := MDDST1([n,n,n]);

name := t.name::"_"::StringInt(n)::"x"::StringInt(n)::"x"::StringInt(n);

t := TFCall(t, rec(fname := name, params := []));

opts := conf.getOpts(t);
opts.breakdownRules.SkewDTT := opts.breakdownRules.SkewDTT{[1..2]};

tt := opts.tagIt(t);
#rt := opts.search(tt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo(name::".c", opts.prettyPrint(c));
cyc := CMeasure(c, opts);

mm := CMatrix(c, opts);
tm := MatSPL(tt);

delta := InfinityNormMat(mm-tm);


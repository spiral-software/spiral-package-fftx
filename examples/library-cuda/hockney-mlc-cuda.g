Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confHockneyMlcCUDADevice();
opts := FFTXGlobals.getOpts(conf);

n := 130;
ns := 33;
nd := 96;

t := let(name := "hockney"::StringInt(n)::"_"::StringInt(nd)::"_"::StringInt(ns), 
        symvar := var("symbl", TPtr(TReal)),
    TFCall(
        Compose([
            ExtractBox([n,n,n], [[n-nd..n-1],[n-nd..n-1],[n-nd..n-1]]),
            IMDPRDFT([n,n,n], 1),
            RCDiag(FDataOfs(symvar, 2*n*n*(n/2+1), 0)),
            MDPRDFT([n,n,n], -1), 
            ZeroEmbedBox([n,n,n], [[0..ns-1],[0..ns-1],[0..ns-1]])]),
        rec(fname := name, params := [symvar])
    ).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);


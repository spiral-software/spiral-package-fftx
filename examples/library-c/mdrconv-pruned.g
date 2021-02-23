
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# nD IO Pruned Real Convolution (aka Hockney)
# only 3D case works, for other situations the necessary breakdown rule is missing

Load(fftx);
ImportAll(fftx);

# use the configuration for small mutidimensional real convolutions
# later we will have to auto-derive the correct options class

conf := LocalConfig.fftx.defaultConf();

d := 3;
n := 128;
ns := n/2;
nd := n/2;
name := "prconv"::StringInt(d);

PrintLine("mdrconv-pruned: n = ", n, " nd = ", nd, " ns = ", ns, ";\t\t##PICKME##");

t := let(name := name, 
        symvar := var("symvar", TPtr(TReal)),
    TFCall(
        Compose([
            ExtractBox(Replicate(d, n), Replicate(d, [n-nd..n-1])),
            IMDPRDFT(Replicate(d, n), 1),
            RCDiag(FDataOfs(symvar, 2*n^(d-1)* (n/2+1), 0)),
            MDPRDFT(Replicate(d, n), -1), 
            ZeroEmbedBox(Replicate(d, n), Replicate(d, [0..ns-1]))]),
        rec(fname := name, params := [symvar])
    )
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);
PrintTo("name"::".c", opts.prettyPrint(c));

PrintLine("mdrconv-pruned: codegen test only (no compiled test with 'symbol')\t\t##PICKME##");

# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

d := 3;
szcube := 80;

PrintLine("mdprdft-cuda: d = ", d, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(ns := Replicate(3, szcube),
    name := "mdprdft"::StringInt(d)::"d", 
#    TFCall(TTensorI(MDPRDFT(ns, 1), 1, AVec, AVec), 
    TFCall(MDPRDFT(ns, 1), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo("imdprdft80b.c", opts.prettyPrint(c));



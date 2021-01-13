# 1d and multidimensional real iDFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

d := 3;
szcube := 4;
name := "imdprdft"::StringInt(d)::"d";

PrintLine("imdprdft-cuda: d = ", d, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(ns := Replicate(3, szcube),
    TFCall(IMDPRDFT(ns, 1), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo(name::".c", opts.prettyPrint(c));



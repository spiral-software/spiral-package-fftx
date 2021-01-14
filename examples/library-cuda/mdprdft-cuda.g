# 1d and multidimensional real DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

d := 3;
szcube := 4;
name := "mdprdft"::StringInt(d)::"d";

PrintLine("mdprdft-cuda: d = ", d, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(ns := Replicate(3, szcube),
    TFCall(MDPRDFT(ns, 1), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo(name::".c", opts.prettyPrint(c));



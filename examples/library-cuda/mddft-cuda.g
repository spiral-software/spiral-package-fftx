# 1d and multidimensional complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

d := 3;
szcube := 4;
name := "mddft"::StringInt(d)::"d";

PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, "^3;\t\t##PICKME##");

t := let(ns := Replicate(3, szcube),
    TFCall(MDDFT(ns, 1), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo(name::".c", opts.prettyPrint(c));



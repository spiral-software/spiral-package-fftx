Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultHIPConf();
N := 256;
t := let(
    name := "grid_dft"::"d_cont",
    TFCall(TTensorI(PRDFT1(N, -1), N*N, APar, AVec), 
        rec(fname := name, params := []))
);
opts := conf.getOpts(t);
tt := opts.tagIt(t);
c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo("grid_dft"::"d_cont1rc"::".c", opts.prettyPrint(c));


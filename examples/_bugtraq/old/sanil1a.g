Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultHIPConf();
N := 256;
t := let(
    name := "grid_dft"::"d_cont",
    TFCall(TRC(TTensorI(DFT(N, -1), N*N, APar, APar)), 
        rec(fname := name, params := []))
);
opts := conf.getOpts(t);
tt := opts.tagIt(t);
c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo("grid_dft"::"d_cont1a"::".c", opts.prettyPrint(c));


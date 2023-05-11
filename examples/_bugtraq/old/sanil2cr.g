Load(fftx);
ImportAll(fftx);
ImportAll(realdft);

conf := FFTXGlobals.defaultHIPConf();
N := 256;
t := let(
    name := "grid_dft"::"d_cont",
    TFCall(TTensorI(IPRDFT(N, -1), N*N, AVec, APar), 
        rec(fname := name, params := []))
);
opts := conf.getOpts(t);
tt := opts.tagIt(t);
c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo("grid_dft"::"d_cont2cr"::".c", opts.prettyPrint(c));


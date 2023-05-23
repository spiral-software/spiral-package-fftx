Load(fftx);
ImportAll(fftx);

#conf := FFTXGlobals.defaultHIPConf();
conf := LocalConfig.fftx.confGPU();

N1 := 32;
N := 2;
t := let(
    name := "grid_dft"::"d_cont",
    TFCall(TRC(TTensorI(DFT(N1, -1), N*N, APar, APar)), 
        rec(fname := name, params := []))
);
opts := conf.getOpts(t);
tt := opts.tagIt(t);
c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo("grid_dft"::"d_cont1a"::".c", opts.prettyPrint(c));

cyc := CMeasure(c, opts);
mm := CVector(c, [1], opts);


mm := CMatrix(c, opts);
m2 := MatSPL(t);
InfinityNormMat(m2-mm);


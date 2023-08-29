Load(fftx);
ImportAll(fftx);

# M = 64;
# N = 64;
# K = 64;
# Batch = 4;
# 
# This gives values to rocfft for any one dimension with APar, APar configuration:
# N = 64;
# istride= 4;
# idist = 256 (64*4, the dft length times the batch size);
# ostride = 4;
# odist = 256 (64*4);
# batch = 4096 (the lengths of the other 2 dimensions);



#conf := FFTXGlobals.defaultHIPConf();
conf := LocalConfig.fftx.confGPU();

N1 := 64;

# works
#N := 2;
N := 64*64;
# testing
Nb := 4;

# reaL size
#N := 256;
#N1 := 256;

# works
pat1 := APar; pat2 := APar;
# works
#pat1 := AVec; pat2 := AVec;
# works
#pat1 := APar; pat2 := AVec;
# works
#pat1 := AVec; pat2 := APar;

t := let(
    name := "grid_dft"::"d_cont",
    TFCall(TRC(TTensorI(TTensorI(DFT(N1, -1), N, pat1, pat2), Nb, AVec, AVec)), 
        rec(fname := name, params := []))
);
opts := conf.getOpts(t);

tt := opts.tagIt(t);
c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo("grid_dft"::"d_cont1a"::".c", opts.prettyPrint(c));

cyc := CMeasure(c, opts);

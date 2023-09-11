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

# target
# N1 := 64;
# N := 64*64;
# Nb := 4;

# test
N1 := 64;
# up to N=16 CMatrix verification is ok
N := 2048;
Nb := 4;



# works
pat1 := APar; pat2 := APar;

# not tested
#pat1 := AVec; pat2 := AVec;
#pat1 := APar; pat2 := AVec;
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

## -- from here on only for smallsizes
# mm := CMatrix(c, opts);;
# m2 := MatSPL(t);;
# InfinityNormMat(m2-mm);
# 


#==
i := 0;
n := N1*N*Nb*2;
#i := Random([0..n-1]);
v := BasisVec(n, i);;
mv := CVector(c, v, opts);;
mv2 := Flat(List([1..N1], i->[1.0, 0.0] :: Replicate(2 * (Nb-1), 0.0)))::Replicate(2 * (N-1) * N1* Nb, 0.0);
InfinityNormMat([mv-mv2]);



#mv2 := List(mm, j->j[i+1]);;
#InfinityNormMat([mv] - [mv2]);



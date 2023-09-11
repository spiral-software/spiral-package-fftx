Load(fftx);
ImportAll(fftx);
ImportAll(realdft);

#conf := FFTXGlobals.defaultHIPConf();
conf := LocalConfig.fftx.confGPU();

N1 := 32;

# testing
N := 2;
#N := 4;
#N := 8;

# reaL size
#N := 256;
#N1 := 256;

# testing
#pat1 := APar; pat2 := APar;
pat1 := AVec; pat2 := APar;

t := let(
    name := "grid_dft"::"d_cont",
    TFCall(TTensorI(IPRDFT(N1, -1), N*N, pat1, pat2), 
        rec(fname := name, params := []))
);
opts := conf.getOpts(t);
tt := opts.tagIt(t);
c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo("grid_dft"::"d_cont2cr"::".c", opts.prettyPrint(c));

cyc := CMeasure(c, opts);
## -- from here on only for smallsizes

mm := CMatrix(c, opts);;
m2 := MatSPL(t);;
InfinityNormMat(m2-mm);

#i := 1;
n := Length(m2);
i := Random([0..n-1]);
v := BasisVec(n, i);;
mv := CVector(c, v, opts);;
mv2 := List(mm, j->j[i+1]);;
InfinityNormMat([mv] - [mv2]);



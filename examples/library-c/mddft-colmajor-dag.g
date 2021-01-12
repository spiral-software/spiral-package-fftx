# 2D/3D and multidimensional complex DFTs in column major (Fortran) data layout

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

#szns := [2, 3];
szns := [2, 3, 4];

Xcmj := tcast(TPtr(TColMaj(BoxND(szns, TComplex))), X);
Ycmj := tcast(TPtr(TColMaj(BoxND(szns, TComplex))), Y);

PrintLine("mddft_cmj: ns = ", szns, ";\t\t##PICKME##");

t := let(ns := szns,
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_cmj",  
    TFCall(
        TDAG([
            TDAGNode(TRC(MDDFT(ns, k)), Ycmj, Xcmj)
        ]), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



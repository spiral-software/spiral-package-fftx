
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 2D/3D and multidimensional complex DFTs in column major (Fortran) data layout

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.defaultConf();

#szns := [2, 3];
szns := [2, 3, 4];

# FIXME: THIS IS NOT YET WORKING
Xcmj := tcast(TPtrLayout(TComplex, BoxNDcmaj(szns)), X);
Ycmj := tcast(TPtrLayout(TComplex, BoxNDcmaj(szns)), Y);

PrintLine("mddft_cmj: ns = ", szns, ";\t\t##PICKME##");

t := let(ns := szns,
    k := -1,
    name := "dft"::StringInt(Length(ns))::"d_cmj",  
    TFCall(
        TDAG([
            TDAGNode(TRC(MDDFT(ns, k)), Ycmj, Xcmj)
        ]), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);




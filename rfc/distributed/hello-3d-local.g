# 3D and multidimensional complex DFTs with cube rotation
# Example usage: local FFTX kernel for global FFT

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.FFTXLocalConf();
opts := FFTXGlobals.getOpts(conf);

Ns := [80, 80, 80];
k := -1;
X_linorder := [dimX, dimZ, dimY];
Y_linorder := [dimY, dimX, dimZ1];

Xrot := tcast(TPtr(LinBoxND(TComplex, X_linorder, Ns)), X);
Yrot := tcast(TPtr(LinBoxND(TComplex, Y_linorder, Ns)), Y);

t := let(
    name := "mddft"::StringInt(Length(ns))::"d_rot",  
    TFCall(
        TDAG([
            TDAGNode(TRC(MDDFT(Ns, k)), Yrot, Xrot)
        ]), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



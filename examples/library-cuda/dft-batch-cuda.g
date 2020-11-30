# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);

n := 2;
N := 2;
xdim := n;
ydim := n;
zdim := n;
ix := Ind(xdim);
iy := Ind(ydim);
iz := Ind(zdim);

t := let(
    name := "grid_dft",
    TFCall(TRC(TMap(DFT(N, -1), [iz, iy, ix], AVec, AVec)), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



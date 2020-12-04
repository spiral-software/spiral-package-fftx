# 3d batch of 1d real DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := FFTXGlobals.confFFTCUDADevice();
opts := FFTXGlobals.getOpts(conf);


n := 4;
N := 4;
xdim := n;
ydim := n;
zdim := n;
ix := Ind(xdim);
iy := Ind(ydim);
iz := Ind(zdim);

PrintLine("prdft-batch-cuda: X/Y/Z dim = ", n, " N = ", N, ";\t\t##PICKME##");

t := let(
    name := "grid_prdft",
    TFCall(TMap(PRDFT(N, -1), [iz, iy, ix], APar, APar), 
        rec(fname := name, params := [])).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);



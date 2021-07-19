
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 3d batch of 1d real DFTs

Load(fftx);
ImportAll(fftx);
Import(realdft);

conf := LocalConfig.fftx.confGPU();

n := 8;
N := 128;

xdim := n;
ydim := n;
zdim := n;
ix := Ind(xdim);
iy := Ind(ydim);
iz := Ind(zdim);

pdft := When(true,
    PRDFT,
    IPRDFT
);

PrintLine("prdft-batch-cuda: X/Y/Z dim = ", n, " N = ", N, ";\t\t##PICKME##");

t := let(
    name := "grid_prdft",
    TFCall(TMap(pdft(N, -1), [iz, iy, ix], APar, APar), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);





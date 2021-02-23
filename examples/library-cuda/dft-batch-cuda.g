
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

n := 2;
N := 2;

PrintLine("dft-batch-cuda: X/Y/Z dim = ", n, " N = ", N, ";\t\t##PICKME##");

xdim := n;
ydim := n;
zdim := n;
ix := Ind(xdim);
iy := Ind(ydim);
iz := Ind(zdim);

t := let(
    name := "grid_dft",
    TFCall(TRC(TMap(DFT(N, -1), [iz, iy, ix], AVec, AVec)), 
        rec(fname := name, params := []))
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);


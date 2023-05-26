
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

N := 1024;
batch := 2;
name := "batch_dft_"::StringInt(batch)::"x"::StringInt(N);

t := TFCall(TRC(TTensorI(DFT(N, -1), batch, APar, AVec)), rec(fname := name, params := []));
#t := TFCall(TRC(DFT(N, -1)), rec(fname := name, params := []));

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

cyc := CMeasure(c, opts);

mt := MatSPL(tt);
mc := CMatrix(c, opts);

diff := InfinityNormMat(mc-mt);


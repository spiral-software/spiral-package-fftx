
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

N := 100000; batch := 2;
# N := 768; batch := 768^2;
# N := 16384; batch := 2;
# N := 8192; batch := 2;
# N := 1024; batch := 1024;

# N := 4096*3; batch := 2;


# N := 100000; batch := 2;

# N := 1024; batch := 1024;
# N := 2048; batch := 4096;
# N := 4096; batch := 16384;
# N := 8192; batch := 65536;


 #N := 16384; batch := 16;
# N := 32768; batch := 16;
# N := 68040; batch := 2;
# N := 72250 ; batch := 2;
# N := 65536; batch := 2;
 #N := 65625; batch := 2;
# N := 32768*3; batch := 2;
N := 68040; batch := 2;
 
 
# [ 65450, 65520, 65536, 65625 ]
 

name := "batch_dft_"::StringInt(batch)::"x"::StringInt(N);

t := TFCall(TRC(TTensorI(DFT(N, -1), batch, APar, AVec)), rec(fname := name, params := []));
#t := TFCall(TRC(TTensorI(DFT(N, -1), batch, AVec, APar)), rec(fname := name, params := []));
#t := TFCall(TRC(DFT(N, -1)), rec(fname := name, params := [])); batch := 1;

opts := conf.getOpts(t);
tt := opts.tagIt(t);
 
# _tt := opts.preProcess(tt);
# rt := opts.search(_tt);
# ss := opts.sumsRuleTree(rt);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

# does it run?
cyc := CMeasure(c, opts);

# quick correctness check for large sizes
# get first non-trivial vector
cv := CVector(c, Replicate(2*batch, 0)::[1], opts);
cv1 := cv{[1..Length(cv)/batch]};
cv1a := Flat(List([0..N-1], k -> [re(E(N)^k).v, -im(E(N)^k).v]));

# correctnes test: true and \approx 10^-14 or so
ForAll(cv{[Length(cv)/batch+1..Length(cv)]}, i -> i = 0.0);
InfinityNormMat([cv1] - [cv1a]);

# find the problems...
x := List([1..Length(cv1a)], i->cv1[i] - cv1a[i]);
xa := List(x, i->abs(i).v);
xx := Zip2(xa, [1..Length(xa)]);
y := Filtered(xx, i-> i[1] > 1e-5);
idx := List(y, i->i[2]);

dists := List([1..Length(idx)-1], j-> idx[j+1]-idx[j]);
dd := Set(Filtered(dists, i-> i<> 1));


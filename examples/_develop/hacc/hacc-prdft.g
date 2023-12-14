
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);
ImportAll(realdft);
ImportAll(fftx.platforms.cuda);

Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

# N := 128; batch := 2;
# N := 256; batch := 2;
# N := 512; batch := 2;
# N := 1024; batch := 2;
# N := 1024; batch := 1024;
# N := 2048; batch := 2;
# N := 4096; batch := 16384;
# N := 8192; batch := 65536;
# N := 8192; batch := 2;
# N := 16384; batch := 16;
# N := 32768; batch := 16;
# N := 65536; batch := 16;

# a.	Sizes that build code: 4000, 4250, 5000, 5250, 6250, 7000, 8000, 8750, 9000, 10000, 10500, 11000, 11250
# a.	Sizes that build code:  4608, 5120, 5376, 5632, 6144, 6400, 6656, 6912, 7168, 7680 
# 
# c.	Sizes that fail:  4500, 4750, 5500, 5750, 6000, 6500, 6750, 7250, 7500, 7750, 8250, 8500, 9250, 9500, 9750, 10250, 10750, 11500, 11750
# c.	Sizes that fail:   4352, 4864, 5888, 7424, 7936

sizes := [4500, 4750, 5500, 5750, 6000, 6500, 6750, 7250, 7500, 7750, 8250, 8500, 9250, 9500, 9750, 10250, 10750, 11500, 11750, 4352, 4864, 5888, 7424, 7936];
sizes := Filtered(sizes, i -> Length(factorize(i, 26, 13)) = 3);


N := sizes[5]; batch := 2;

name := "batch_dft_"::StringInt(batch)::"x"::StringInt(N);

#t := TFCall(TRC(TTensorI(PRDFT1(N, -1), batch, APar, APar)), rec(fname := name, params := []));
t := TFCall(TRC(TTensorI(IPRDFT1(N, -1), batch, APar, APar)), rec(fname := name, params := []));

opts := conf.getOpts(t);
tt := opts.tagIt(t);
 
## ==
#_tt := opts.preProcess(tt);
#rt := opts.search(_tt);
# ss := opts.sumsRuleTree(rt);
# c := opts.codeSums(ss);
# 
c := opts.fftxGen(tt);
opts.prettyPrint(c);

cyc := CMeasure(c, opts);

# mt := MatSPL(tt);
# mc := CMatrix(c, opts);
# 
# diff := InfinityNormMat(mc-mt);
# 
# # =====
# 
# 
# 
# # 
# # c := opts.fftxGen(tt);
# # opts.prettyPrint(c);
# 
# # does it run?
# cyc := CMeasure(c, opts);
# 
# # quick correctness check for large sizes
# # get first non-trivial vector
# cv := CVector(c, Replicate(2*batch, 0)::[1], opts);
# cv1 := cv{[1..Length(cv)/batch]};
# cv1a := Flat(List([0..N-1], k -> [re(E(N)^k).v, -im(E(N)^k).v]));
# 
# # correctnes test: true and \approx 10^-14 or so
# ForAll(cv{[Length(cv)/batch+1..Length(cv)]}, i -> i = 0.0);
# InfinityNormMat([cv1] - [cv1a]);
# 
# 
# 

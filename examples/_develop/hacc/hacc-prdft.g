
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


# N := 256; batch := 2;
N := 512; batch := 2;
# N := 1024; batch := 2;
# N := 1024; batch := 1024;
# N := 2048; batch := 4096;
# N := 4096; batch := 16384;
# N := 8192; batch := 65536;

# N := 16384; batch := 16;
# N := 32768; batch := 16;
# N := 65536; batch := 16;

name := "batch_dft_"::StringInt(batch)::"x"::StringInt(N);

t := TFCall(TRC(TTensorI(PRDFT1(N, -1), batch, APar, APar)), rec(fname := name, params := []));
#t := TFCall(TRC(TTensorI(IPRDFT1(N, -1), batch, APar, APar)), rec(fname := name, params := []));

opts := conf.getOpts(t);

MAX_KERNEL := 26;
MAX_PRIME := 17;
opts.globalUnrolling := 64;

opts.breakdownRules.PRDFT[3].allChildren := P -> Filtered(PRDFT1_CT.allChildren(P), 
    e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
        Cond(Length(factors) = 1 , true, Length(factors) <= 3, e[1].params[1] = factors[1], e[1].params[1] = factors[1]*factors[2])));

opts.breakdownRules.IPRDFT[3].allChildren := P -> Filtered(IPRDFT1_CT.allChildren(P), 
    e-> let(factors := factorize(e[1].params[1]*e[3].params[1], MAX_KERNEL, MAX_PRIME), 
        Cond(Length(factors) = 1 , true, Length(factors) <= 3, e[1].params[1] = factors[1], e[1].params[1] = factors[1]*factors[2])));


tt := opts.tagIt(t);

# ==
_tt := opts.preProcess(tt);
rt := opts.search(_tt);

#-- breakdown seems fine, but SumsSPL doesn't do the right thing


ss := opts.sumsRuleTree(rt);


c := opts.codeSums(ss);
# ==

c := opts.fftxGen(tt);
opts.prettyPrint(c);

cyc := CMeasure(c, opts);

mt := MatSPL(tt);
mc := CMatrix(c, opts);

diff := InfinityNormMat(mc-mt);
# 
# c := opts.fftxGen(tt);
# opts.prettyPrint(c);

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



